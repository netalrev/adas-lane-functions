/**
 * cpp/src/adas_inference.cpp
 * ============================
 * Implementation of AdasInference — ONNX Runtime wrapper for MFTransformer.
 */

#include "adas_inference.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace adas {

// ── Constructor ───────────────────────────────────────────────────────────────

AdasInference::AdasInference(const std::string& model_path, int num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "AdasInference"),
      session_opts_(),
      session_(nullptr),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    session_opts_.SetIntraOpNumThreads(num_threads);
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = Ort::Session(env_, model_path.c_str(), session_opts_);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(
            std::string("[AdasInference] Failed to load model: ") + e.what() +
            "\n  Path: " + model_path
        );
    }

    // Cache input / output names from the model metadata.
    Ort::AllocatorWithDefaultOptions alloc;

    auto in_name      = session_.GetInputNameAllocated(0, alloc);
    auto cipv_name    = session_.GetOutputNameAllocated(0, alloc);
    auto lane_name    = session_.GetOutputNameAllocated(1, alloc);
    auto cut_in_name  = session_.GetOutputNameAllocated(2, alloc);

    input_name_        = std::string(in_name.get());
    cipv_output_name_  = std::string(cipv_name.get());
    lane_output_name_  = std::string(lane_name.get());
    cut_in_output_name_= std::string(cut_in_name.get());
}


// ── Inference ─────────────────────────────────────────────────────────────────

std::vector<TrackPrediction> AdasInference::run(const std::vector<TrackInput>& tracks)
{
    if (tracks.empty()) return {};

    const int64_t batch = static_cast<int64_t>(tracks.size());

    // Build flattened [N, T, D] input buffer.
    std::vector<float> input_buf(static_cast<size_t>(batch * T * D));
    for (int64_t b = 0; b < batch; ++b) {
        const float* src = tracks[static_cast<size_t>(b)].mf.data();
        float*       dst = input_buf.data() + b * T * D;
        std::copy(src, src + T * D, dst);
    }

    // Create input tensor.
    const std::array<int64_t, 3> input_shape = {batch, T, D};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_buf.data(),
        input_buf.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Run inference.
    const char* input_names[]  = {input_name_.c_str()};
    const char* output_names[] = {
        cipv_output_name_.c_str(),
        lane_output_name_.c_str(),
        cut_in_output_name_.c_str(),
    };

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor, 1,
        output_names, 3
    );

    // Decode raw logits.
    const float* cipv_logits   = output_tensors[0].GetTensorData<float>();
    const float* lane_logits   = output_tensors[1].GetTensorData<float>();
    const float* cut_in_logits = output_tensors[2].GetTensorData<float>();

    std::vector<TrackPrediction> preds(static_cast<size_t>(batch));

    // Per-track post-processing: sigmoid, softmax, argmax.
    for (int64_t b = 0; b < batch; ++b) {
        auto& pred        = preds[static_cast<size_t>(b)];
        pred.track_id     = tracks[static_cast<size_t>(b)].track_id;

        // CIPV probability (not yet thresholded — CIPV flag set below).
        pred.cipv_prob    = sigmoid(cipv_logits[b]);

        // Lane assignment: softmax over 5 classes, then argmax.
        const float* lane_row = lane_logits + b * N_LANE_CLASSES;
        std::copy(lane_row, lane_row + N_LANE_CLASSES, pred.lane_probs.data());
        softmax(pred.lane_probs.data(), N_LANE_CLASSES);
        int best_cls = static_cast<int>(
            std::max_element(pred.lane_probs.begin(), pred.lane_probs.end())
            - pred.lane_probs.begin()
        );
        pred.lane_assignment = LANE_OFFSETS[static_cast<size_t>(best_cls)];

        // Cut-in probability and flag.
        pred.cut_in_prob = sigmoid(cut_in_logits[b]);
        pred.cut_in      = pred.cut_in_prob >= CUT_IN_THRESHOLD;
    }

    // CIPV is assigned to the single track with the highest probability,
    // provided it exceeds the threshold.  At most one CIPV per frame.
    if (batch > 0) {
        auto best_it = std::max_element(
            preds.begin(), preds.end(),
            [](const TrackPrediction& a, const TrackPrediction& b) {
                return a.cipv_prob < b.cipv_prob;
            }
        );
        if (best_it->cipv_prob >= CIPV_THRESHOLD) {
            best_it->cipv = true;
        }
    }

    return preds;
}


// ── Helpers ───────────────────────────────────────────────────────────────────

float AdasInference::sigmoid(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

void AdasInference::softmax(float* data, int n) noexcept {
    float max_val = *std::max_element(data, data + n);
    float sum     = 0.0f;
    for (int i = 0; i < n; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum    += data[i];
    }
    for (int i = 0; i < n; ++i) data[i] /= sum;
}

}  // namespace adas

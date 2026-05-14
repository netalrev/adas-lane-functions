/**
 * cpp/include/adas_inference.hpp
 * ================================
 * C++ API for MFTransformer ONNX Runtime inference.
 *
 * Provides a single-class interface for per-frame ADAS signal prediction.
 * The caller is responsible for assembling the MF windows (T frames × D features
 * per track) and passes them as a flat vector.  AdasInference batches all tracks
 * in a frame into a single ONNX inference call for efficiency.
 *
 * Input JSON format (used by main.cpp)
 * -------------------------------------
 *   {
 *     "frame_idx": 42,
 *     "tracks": [
 *       {
 *         "track_id": 1,
 *         "mf_window": [[f0,f1,...,f17], ..., [f0,f1,...,f17]]  // T×D
 *       }
 *     ]
 *   }
 *
 * Output JSON format
 * -------------------
 *   {
 *     "frame_idx": 42,
 *     "predictions": [
 *       {
 *         "track_id": 1,
 *         "cipv_prob":       0.87,
 *         "cipv":            true,
 *         "lane_assignment": 0,
 *         "lane_probs":      [0.01, 0.02, 0.95, 0.01, 0.01],
 *         "cut_in_prob":     0.11,
 *         "cut_in":          false
 *       }
 *     ]
 *   }
 *
 * Lane assignment mapping
 * -----------------------
 *   Class index → lane offset relative to ego:
 *     0 → -2  (two lanes left)
 *     1 → -1  (one lane left)
 *     2 →  0  (ego lane)
 *     3 → +1  (one lane right)
 *     4 → +2  (two lanes right)
 *   The "cipv" flag is set for the single track with the highest cipv_prob,
 *   provided it exceeds CIPV_THRESHOLD.
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace adas {

// ── Constants ─────────────────────────────────────────────────────────────────

constexpr int   T             = 10;     ///< Temporal window length (frames).
constexpr int   D             = 18;     ///< Feature vector dimension.
constexpr int   N_LANE_CLASSES = 5;     ///< Number of lane assignment classes.
constexpr float CIPV_THRESHOLD = 0.5f;  ///< Sigmoid threshold for CIPV positive.
constexpr float CUT_IN_THRESHOLD = 0.5f; ///< Sigmoid threshold for cut-in positive.

/// Lane offset values corresponding to each class index (0..4).
constexpr std::array<int, N_LANE_CLASSES> LANE_OFFSETS = {-2, -1, 0, 1, 2};


// ── Data structures ───────────────────────────────────────────────────────────

/// MF window for a single track: T frames × D features (row-major).
using MfWindow = std::array<float, T * D>;

/// Per-track input to the inference engine.
struct TrackInput {
    int64_t  track_id{0};
    MfWindow mf{};      ///< T×D feature window, row-major (frame 0 first).
};

/// Per-track prediction output.
struct TrackPrediction {
    int64_t track_id{0};

    float cipv_prob{0.0f};          ///< Sigmoid probability of CIPV.
    bool  cipv{false};              ///< True for the single highest-prob CIPV track.

    int   lane_assignment{0};       ///< Lane offset in {-2,-1,0,+1,+2}.
    std::array<float, N_LANE_CLASSES> lane_probs{};  ///< Softmax probabilities.

    float cut_in_prob{0.0f};        ///< Sigmoid probability of cut-in.
    bool  cut_in{false};            ///< True if cut_in_prob > threshold.
};


// ── AdasInference ─────────────────────────────────────────────────────────────

/**
 * ONNX Runtime wrapper for the MFTransformer model.
 *
 * Thread safety: each instance owns its own ORT session and is not thread-safe.
 * For multi-threaded use, create one instance per thread.
 */
class AdasInference {
public:
    /**
     * Load the ONNX model from disk.
     *
     * @param model_path  Path to the .onnx file (FP32 or INT8).
     * @param num_threads Number of intra-op threads for ORT (default: 1).
     * @throws std::runtime_error if the model cannot be loaded.
     */
    explicit AdasInference(const std::string& model_path, int num_threads = 1);

    AdasInference(const AdasInference&)            = delete;
    AdasInference& operator=(const AdasInference&) = delete;
    AdasInference(AdasInference&&)                 = default;
    AdasInference& operator=(AdasInference&&)      = default;
    ~AdasInference()                               = default;

    /**
     * Run per-frame inference for all active tracks.
     *
     * Batches all tracks into a single [N, T, D] tensor and runs one ONNX
     * call.  The CIPV flag is assigned to the single track with the highest
     * cipv_prob (provided it exceeds CIPV_THRESHOLD).
     *
     * @param tracks  Per-track MF windows to classify.
     * @return        One TrackPrediction per input TrackInput, in the same order.
     */
    std::vector<TrackPrediction> run(const std::vector<TrackInput>& tracks);

private:
    Ort::Env            env_;
    Ort::SessionOptions session_opts_;
    Ort::Session        session_;
    Ort::MemoryInfo     memory_info_;

    // ORT-managed input/output names (kept alive for the session's lifetime).
    std::string input_name_;
    std::string cipv_output_name_;
    std::string lane_output_name_;
    std::string cut_in_output_name_;

    static float   sigmoid(float x) noexcept;
    static void    softmax(float* data, int n) noexcept;
};

}  // namespace adas

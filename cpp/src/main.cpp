/**
 * cpp/src/main.cpp
 * =================
 * CLI driver for the ADAS inference runtime.
 *
 * Reads a per-frame JSON file containing MF windows for all active tracks,
 * runs ONNX Runtime inference, and writes predictions as JSON to stdout.
 *
 * Usage
 * -----
 *   adas_infer --model <path.onnx> --input <frame.json>
 *   adas_infer --model <path.onnx> --input <frame.json> --threshold 0.6
 *
 * Input JSON schema
 * -----------------
 *   {
 *     "frame_idx": 42,
 *     "tracks": [
 *       { "track_id": 1, "mf_window": [[18 floats] × 10] },
 *       ...
 *     ]
 *   }
 *
 * Output JSON schema (written to stdout)
 * ----------------------------------------
 *   {
 *     "frame_idx": 42,
 *     "predictions": [
 *       {
 *         "track_id":        1,
 *         "cipv_prob":       0.87,
 *         "cipv":            true,
 *         "lane_assignment": 0,
 *         "lane_probs":      [0.01, 0.02, 0.95, 0.01, 0.01],
 *         "cut_in_prob":     0.11,
 *         "cut_in":          false
 *       }
 *     ]
 *   }
 */

#include "adas_inference.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Argument parsing ──────────────────────────────────────────────────────────

struct Args {
    std::string model_path;
    std::string input_path;
    int         threads{1};
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            a.model_path = argv[++i];
        } else if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
            a.input_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            a.threads = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: adas_infer --model <path.onnx> --input <frame.json> "
                "[--threads N]\n";
            std::exit(0);
        }
    }
    if (a.model_path.empty() || a.input_path.empty()) {
        std::cerr << "[adas_infer] Error: --model and --input are required.\n"
                  << "  Usage: adas_infer --model <path.onnx> --input <frame.json>\n";
        std::exit(1);
    }
    return a;
}


// ── JSON I/O helpers ──────────────────────────────────────────────────────────

static json load_json(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open input file: " + path);
    }
    return json::parse(ifs);
}

static std::vector<adas::TrackInput> parse_tracks(const json& frame_json) {
    std::vector<adas::TrackInput> tracks;

    for (const auto& t : frame_json.at("tracks")) {
        adas::TrackInput ti;
        ti.track_id = t.at("track_id").get<int64_t>();

        const auto& win = t.at("mf_window");
        if (win.size() != adas::T) {
            throw std::runtime_error(
                "Track " + std::to_string(ti.track_id) +
                ": mf_window has " + std::to_string(win.size()) +
                " rows, expected " + std::to_string(adas::T)
            );
        }
        for (int row = 0; row < adas::T; ++row) {
            const auto& feat = win[static_cast<size_t>(row)];
            if (feat.size() != adas::D) {
                throw std::runtime_error(
                    "Track " + std::to_string(ti.track_id) +
                    " row " + std::to_string(row) +
                    ": expected " + std::to_string(adas::D) + " features, "
                    "got " + std::to_string(feat.size())
                );
            }
            for (int d = 0; d < adas::D; ++d) {
                ti.mf[static_cast<size_t>(row * adas::D + d)] =
                    feat[static_cast<size_t>(d)].get<float>();
            }
        }
        tracks.push_back(std::move(ti));
    }
    return tracks;
}

static json predictions_to_json(
    const std::vector<adas::TrackPrediction>& preds,
    int64_t frame_idx
) {
    json out;
    out["frame_idx"]    = frame_idx;
    out["predictions"]  = json::array();

    for (const auto& p : preds) {
        json entry;
        entry["track_id"]        = p.track_id;
        entry["cipv_prob"]       = p.cipv_prob;
        entry["cipv"]            = p.cipv;
        entry["lane_assignment"] = p.lane_assignment;
        entry["lane_probs"]      = p.lane_probs;
        entry["cut_in_prob"]     = p.cut_in_prob;
        entry["cut_in"]          = p.cut_in;
        out["predictions"].push_back(entry);
    }
    return out;
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);

        // Load model.
        adas::AdasInference engine(args.model_path, args.threads);

        // Load input frame JSON.
        const json  frame_json = load_json(args.input_path);
        const int64_t frame_idx = frame_json.value("frame_idx", int64_t{0});

        // Parse tracks.
        const auto tracks = parse_tracks(frame_json);
        if (tracks.empty()) {
            json empty_out;
            empty_out["frame_idx"]   = frame_idx;
            empty_out["predictions"] = json::array();
            std::cout << empty_out.dump(2) << "\n";
            return 0;
        }

        // Run inference.
        const auto preds = engine.run(tracks);

        // Write output JSON to stdout.
        std::cout << predictions_to_json(preds, frame_idx).dump(2) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "[adas_infer] Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

// cpp/python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>            // std::vector, std::map 자동 변환
#include <pybind11/functional.h>     // std::function 콜백
#include <nlohmann/json.hpp>         // JSON ↔ Python dict 변환용

#include "recognition_engine.h"

namespace py = pybind11;
using json = nlohmann::json;

PYBIND11_MODULE(pyrealtime, m) {
    m.doc() = "Realtime Korean speech evaluation engine";

    //--- RecordListener 바인딩 ---
    py::class_<realtime_engine_ko::RecordListener>(m, "RecordListener")
        .def(py::init<
            realtime_engine_ko::RecordListener::StartCallback,
            realtime_engine_ko::RecordListener::TickCallback,
            realtime_engine_ko::RecordListener::FailCallback,
            realtime_engine_ko::RecordListener::EndCallback,
            realtime_engine_ko::RecordListener::ScoreCallback
        >(),
        py::arg("on_start") = nullptr,
        py::arg("on_tick") = nullptr,
        py::arg("on_start_record_fail") = nullptr,
        py::arg("on_record_end") = nullptr,
        py::arg("on_score") = nullptr
        )
        // 실제로 Python 함수가 들어올 때 std::function으로 자동 래핑됩니다.
        ;

    //--- EngineCoordinator 바인딩 ---
    py::class_<realtime_engine_ko::EngineCoordinator>(m, "EngineCoordinator")
        .def(py::init<
            const std::string&,
            const std::string&,
            const std::string&,
            float,
            float>(),
             py::arg("onnx_model_path"),
             py::arg("tokenizer_path"),
             py::arg("device") = "CPU",
             py::arg("update_interval") = 0.3f,
             py::arg("confidence_threshold") = 0.7f
        )
        .def("SetRecordListener", &realtime_engine_ko::EngineCoordinator::SetRecordListener)
        .def("Initialize", &realtime_engine_ko::EngineCoordinator::Initialize,
             py::arg("sentence"),
             py::arg("audio_polling_interval") = 0.03f,
             py::arg("min_time_between_evals") = 0.5f
        )
        .def("StartEvaluation", &realtime_engine_ko::EngineCoordinator::StartEvaluation,
             py::arg("audio_file_path"))
        .def("StopEvaluation", &realtime_engine_ko::EngineCoordinator::StopEvaluation)
        // C++ 의 std::map<string, any> 를 Python dict 로 바꾸려면 JSON 을 중간에 씁니다.
        .def("GetCurrentState", [](const realtime_engine_ko::EngineCoordinator &self) {
            auto st = self.GetCurrentState();          // std::map<string, any>
            json j = json::object();
            for (auto &kv : st) {
                // 가정: 모든 값이 string 으로 변환 가능하다면
                j[kv.first] = std::any_cast<std::string>(kv.second);
            }
            return py::cast(j);
        })
        .def("Reset", &realtime_engine_ko::EngineCoordinator::Reset)
        // EvaluateSpeech → 같은 JSON trick
        .def("EvaluateSpeech", [](realtime_engine_ko::EngineCoordinator &self,
                                   const std::string &sent,
                                   const std::string &afile,
                                   const realtime_engine_ko::RecordListener &rl) {
            auto st = self.EvaluateSpeech(sent, afile, rl);
            json j = json::object();
            for (auto &kv : st)
                j[kv.first] = std::any_cast<std::string>(kv.second);
            return py::cast(j);
        },
        py::arg("sentence"),
        py::arg("audio_file_path"),
        py::arg("record_listener") = realtime_engine_ko::RecordListener()
        )
        ;
}

// cpp/python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>            // std::vector, std::map 자동 변환
#include <pybind11/functional.h>     // std::function 콜백
#include <pybind11/eigen.h>          // Eigen::Matrix 변환
#include <nlohmann/json.hpp>         // JSON ↔ Python dict 변환용

#include "realtime_engine_ko/recognition_engine.h"

namespace py = pybind11;
using json = nlohmann::json;

// std::map<std::string, std::any>를 Python 딕셔너리로 변환
py::dict convert_any_map_to_dict(const std::map<std::string, std::any>& map) {
    py::dict result;
    
    for (const auto& [key, value] : map) {
        try {
            if (value.type() == typeid(int)) {
                result[key.c_str()] = std::any_cast<int>(value);
            } else if (value.type() == typeid(float)) {
                result[key.c_str()] = std::any_cast<float>(value);
            } else if (value.type() == typeid(double)) {
                result[key.c_str()] = std::any_cast<double>(value);
            } else if (value.type() == typeid(bool)) {
                result[key.c_str()] = std::any_cast<bool>(value);
            } else if (value.type() == typeid(std::string)) {
                result[key.c_str()] = std::any_cast<std::string>(value);
            } else if (value.type() == typeid(std::vector<std::map<std::string, std::any>>)) {
                auto vec = std::any_cast<std::vector<std::map<std::string, std::any>>>(value);
                py::list py_list;
                for (const auto& item : vec) {
                    py_list.append(convert_any_map_to_dict(item));
                }
                result[key.c_str()] = py_list;
            } else if (value.type() == typeid(std::map<std::string, std::any>)) {
                auto nested_map = std::any_cast<std::map<std::string, std::any>>(value);
                result[key.c_str()] = convert_any_map_to_dict(nested_map);
            }
        } catch (const std::exception& e) {
            py::print("변환 오류:", e.what());
        }
    }
    
    return result;
}

PYBIND11_MODULE(pyrealtime, m) {
    m.doc() = "Realtime Korean speech evaluation engine";

    // RecordListener 클래스는 제거됨 (더 이상 사용하지 않음)
    
    //--- EngineCoordinator 바인딩 ---
    py::class_<realtime_engine_ko::EngineCoordinator>(m, "EngineCoordinator")
        .def(py::init<
            const std::string&,
            const std::string&,
            const std::string&,
            float>(),
             py::arg("onnx_model_path"),
             py::arg("tokenizer_path"),
             py::arg("device") = "CPU",
             py::arg("confidence_threshold") = 0.7f
        )
        .def("create_session", [](realtime_engine_ko::EngineCoordinator &self, 
                               const std::string &sentence,
                               const py::dict &options = py::dict()) {
            std::map<std::string, std::any> cpp_options;
            
            // Python 딕셔너리를 std::map<std::string, std::any>로 변환
            if (!options.empty()) {
                for (auto item : options) {
                    std::string key = py::cast<std::string>(item.first);
                    // 올바른 타입 변환 (handle -> object)
                    py::handle h = item.second;
                    
                    if (py::isinstance<py::float_>(h)) {
                        cpp_options[key] = py::cast<float>(h);
                    } else if (py::isinstance<py::int_>(h)) {
                        cpp_options[key] = py::cast<int>(h);
                    } else if (py::isinstance<py::bool_>(h)) {
                        cpp_options[key] = py::cast<bool>(h);
                    } else if (py::isinstance<py::str>(h)) {
                        cpp_options[key] = py::cast<std::string>(h);
                    }
                }
            }
            
            auto result = self.CreateSession(sentence, cpp_options);
            return convert_any_map_to_dict(result);
        }, py::arg("sentence"), py::arg("options") = py::dict())
        
        .def("evaluate_audio", [](realtime_engine_ko::EngineCoordinator &self,
                               const std::string &session_id,
                               const py::bytes &binary_data) {
            // Python bytes를 std::vector<uint8_t>로 변환
            py::buffer_info buffer = py::buffer(binary_data).request();
            std::vector<uint8_t> cpp_data(static_cast<uint8_t*>(buffer.ptr),
                                      static_cast<uint8_t*>(buffer.ptr) + buffer.size);
            
            auto result = self.EvaluateAudio(session_id, cpp_data);
            return convert_any_map_to_dict(result);
        }, py::arg("session_id"), py::arg("binary_data"))
        
        .def("close_session", [](realtime_engine_ko::EngineCoordinator &self,
                              const std::string &session_id) {
            auto result = self.CloseSession(session_id);
            return convert_any_map_to_dict(result);
        }, py::arg("session_id"))
        
        .def("get_session_status", [](realtime_engine_ko::EngineCoordinator &self,
                                   const std::string &session_id) {
            auto result = self.GetSessionStatus(session_id);
            return convert_any_map_to_dict(result);
        }, py::arg("session_id"))
        
        .def("cleanup_inactive_sessions", &realtime_engine_ko::EngineCoordinator::CleanupInactiveSessions,
             py::arg("max_inactive_time") = 3600.0f)
        ;
}

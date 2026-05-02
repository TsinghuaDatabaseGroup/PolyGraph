//
// Created by mengtong-x on 2026/04/30.
//

#ifndef WEAVESS_PARAMETERS_H
#define WEAVESS_PARAMETERS_H

#include <sstream>
#include <unordered_map>
#include <iostream>

namespace weavess {
    class Parameters {
    public:
        template<typename T>
        inline void set(const std::string &name, const T &val) {
            std::stringstream ss;
            ss << val;
            params[name] = ss.str();
        }

        template<typename T>
        inline T get(const std::string &name) const {
            auto item = params.find(name);
            if (item == params.end()) {
                throw std::invalid_argument("Invalid paramter name : " + name + ".");
            } else {
                return ConvertStrToValue<T>(item->second);
            }
        }

        inline bool exist(const std::string &name) const {
            auto item = params.find(name);
            if (item == params.end()) {
                return false;
            } else {
                return true;
            }
        }

        inline std::string toString() const {
            std::string res;
            for (auto &param : params) {
                res += param.first;
                res += ":";
                res += param.second;
                res += " ";
            }
            return res;
        }

        void showAllParams() {
            std::cout << "【 All Known parameters now are: 】" << std::endl;
            for (const auto& pair : params) {
                std::cout << "      " << pair.first << ": " << pair.second << std::endl;
            }
            std::cout << "【 End 】\n" << std::endl;
        }

    private:
        std::unordered_map<std::string, std::string> params;

        template<typename T>
        inline T ConvertStrToValue(const std::string &str) const {
            std::stringstream sstream(str);
            T value;
            if (!(sstream >> value) || !sstream.eof()) {
                std::stringstream err;
                err << "Fail to convert value" << str << " to type: " << typeid(value).name();
                throw std::runtime_error(err.str());
            }

            return value;
        }
    };
}

#endif //WEAVESS_PARAMETERS_H

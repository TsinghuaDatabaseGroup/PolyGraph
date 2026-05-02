#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>


// C++14: 用 POSIX 方式复制文件并覆盖目标
static int copy_file_overwrite(const std::string& src, const std::string& dst) {
    int in_fd = ::open(src.c_str(), O_RDONLY);
    if (in_fd == -1) {
        return -1;
    }

    struct stat st;
    if (::fstat(in_fd, &st) == -1) {
        int e = errno;
        ::close(in_fd);
        errno = e;
        return -1;
    }

    // 目标文件：覆盖（TRUNC），权限尽量沿用源文件的 rwx 位
    mode_t mode = static_cast<mode_t>(st.st_mode & 0777);
    int out_fd = ::open(dst.c_str(), O_WRONLY | O_CREAT | O_TRUNC, mode);
    if (out_fd == -1) {
        int e = errno;
        ::close(in_fd);
        errno = e;
        return -1;
    }

    // 复制循环
    const size_t BUF_SIZE = 1 << 20; // 1MB
    std::vector<char> buf(BUF_SIZE);

    while (true) {
        ssize_t r = ::read(in_fd, buf.data(), buf.size());
        if (r == 0) break;                 // EOF
        if (r < 0) {                       // read error
            int e = errno;
            ::close(in_fd);
            ::close(out_fd);
            errno = e;
            return -1;
        }

        ssize_t written = 0;
        while (written < r) {
            ssize_t w = ::write(out_fd, buf.data() + written, static_cast<size_t>(r - written));
            if (w < 0) {
                int e = errno;
                ::close(in_fd);
                ::close(out_fd);
                errno = e;
                return -1;
            }
            written += w;
        }
    }

    ::close(in_fd);
    ::close(out_fd);
    return 0;
}

int run_python(
    bool override = true,
    bool rela_use_intersect = true,
    float total_sim_thresh = 0.95, float rela_sim_thresh = 0.5,
    unsigned max_group = 0,
    const std::string& script_path = "../include/python_file/cluster_Now_distSimThresh.py",
    std::string log_path = "../include/python_file/nohup_logs/bash.log",
    const std::string& venv_python = "../pythonEnv_ForSI",
    const std::string& txt_path = "../dataset/path_info.txt",
    const std::vector<std::string>& extra_args = {}
) {
    /**
     * python需要处理的两种指令：
     *      nohup /usr/bin/time -v python3 ../include/python_file/cluster_Now_distSimThresh-relaUseThresh-SyntheticStand.py --total_sim_thresh 0.95 --no-rela_use_intersect --rela_sim_thresh 0.5 --max_group 25 > ../include/python_file/nohup_logs/bash.log51022-clusterSyntheticStand-relaThresh_distSimThresh_0.95 2>&1 
     *      nohup /usr/bin/time -v python3 ../include/python_file/cluster_Now_distSimThresh-relaUseThresh-SyntheticStand.py --total_sim_thresh 0.95 --rela_use_intersect --max_group 25 > ../include/python_file/nohup_logs/bash.log51022-clusterSyntheticStand-useIntersect_distSimThresh_0.95 2>&1 &
     */
    
    std::cout << "___ RUN PYTHON ___" << std::endl;
    std::cout << "-- python file at: " << script_path << std::endl;

    // 0) log-path
    std::ostringstream date_ss;
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    date_ss << std::put_time(std::localtime(&t), "%Y%m%d-%H:%M:%S");  // 形如 20251024-21:40:38

    log_path += date_ss.str();
    log_path += "-clusterNow_distSimThresh_";

    // 1) 组装要传给 python 的参数（-u 便于实时日志）
    std::vector<std::string> py_args;
    py_args.emplace_back("-u");
    py_args.emplace_back(script_path);

    // --txt_path
    py_args.emplace_back("--txt_path");
    py_args.emplace_back(txt_path);

    // --total_sim_thresh
    py_args.emplace_back("--total_sim_thresh");
    py_args.emplace_back(std::to_string(total_sim_thresh));

    if (rela_use_intersect) {
        py_args.emplace_back("--rela_use_intersect");
        log_path += "RelaIntersect_Total" + std::to_string(total_sim_thresh);
    } else {
        py_args.emplace_back("--no-rela_use_intersect");
        py_args.emplace_back("--rela_sim_thresh");
        py_args.emplace_back(std::to_string(rela_sim_thresh));
        log_path += "RelaThresh_Total" + std::to_string(total_sim_thresh) +
                    "_Rela" + std::to_string(rela_sim_thresh);
    }

    // --max_group
    py_args.emplace_back("--max_group");
    py_args.emplace_back(std::to_string(max_group));

    // 追加用户自定义参数
    for (const auto& a : extra_args) py_args.emplace_back(a);

    // 2) 打开日志（追加/覆盖）
    std::cout << "__ Python Log will save to: " << log_path << " __" << std::endl;
    std::cout << "rela_use_intersect: " << rela_use_intersect << std::endl;
    std::cout << "total_sim_thresh: " << total_sim_thresh << std::endl;
    std::cout << "rela_sim_thresh: " << rela_sim_thresh << std::endl;

    int fd = ::open(log_path.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0644);
    if (override) {
        if (fd != -1) ::close(fd);
        fd = ::open(log_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    }
    if (fd == -1) {
        std::perror("open log");
        return 1;
    }

    // 3) fork 子进程执行 /usr/bin/time -v <venv_python> <py_args...>
    pid_t pid = ::fork();
    if (pid < 0) {
        std::perror("fork");
        ::close(fd);
        return 1;
    }

    if (pid == 0) {
        // 子进程：重定向 stdout/stderr 到 log
        if (::dup2(fd, STDOUT_FILENO) == -1) std::perror("dup2 stdout");
        if (::dup2(fd, STDERR_FILENO) == -1) std::perror("dup2 stderr");
        ::close(fd);

        // stdin -> /dev/null
        int nullfd = ::open("/dev/null", O_RDONLY);
        if (nullfd != -1) { ::dup2(nullfd, STDIN_FILENO); ::close(nullfd); }

        const char* prog = "/usr/bin/time";

        // argv: /usr/bin/time -v <venv_python> [py_args...]
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(prog));
        argv.push_back(const_cast<char*>("-v"));
        argv.push_back(const_cast<char*>(venv_python.c_str()));
        for (auto& s : py_args) argv.push_back(const_cast<char*>(s.c_str()));
        argv.push_back(nullptr);

        ::execv(prog, argv.data());
        std::fprintf(stderr, "execv failed: %s\n", std::strerror(errno));
        _exit(127);
    }

    // 父进程：等待 python 结束
    ::close(fd);
    int status = 0;
    while (true) {
        if (::waitpid(pid, &status, 0) == -1) {
            if (errno == EINTR) continue;
            std::perror("waitpid");
            return 1;
        }
        break;
    }

    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        std::cout << "Python job exited with code " << code << "\n";

        // === 成功才复制到 bash.log_now（覆盖） ===
        if (code == 0) {
            const std::string latest_log =
                "../include/python_file/bash.log_now";

            if (copy_file_overwrite(log_path, latest_log) == 0) {
                std::cout << "Log copied to: " << latest_log << "\n";
            } else {
                std::cerr << "copy_file_overwrite failed: " << std::strerror(errno) << "\n";
                // 如果你希望“复制失败也算失败”，可以改成：return 1;
            }
        }

        return code;
    } else if (WIFSIGNALED(status)) {
        std::cout << "Python job killed by signal " << WTERMSIG(status) << "\n";
        return 128 + WTERMSIG(status);
    }

    return 1;
}

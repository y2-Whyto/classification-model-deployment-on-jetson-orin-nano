// ref https://blog.csdn.net/bobchen1017/article/details/129900569
// ref https://blog.csdn.net/qq_41263444/article/details/138301510
// ref https://blog.csdn.net/weixin_38241876/article/details/133177813
// ref https://blog.csdn.net/qq_26611129/article/details/132738109

//#define MY_DEBUG
#define DEBUG_INPUT_H 224  // input height
#define DEBUG_INPUT_W 224  // input width
#define DEBUG_BATCH_SIZE 1  // batch size
#define DEBUG_N_CAT 1000  // sum of categories

#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>  // clang++; for g++ use <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <queue>
#include <utility>
#include <numeric>
#include "include/json.hpp"  // JSON parser lib

//using namespace nvinfer1;
namespace fs = std::experimental::filesystem;  // clang++; for g++ use std::filesystem
using json = nlohmann::json;

std::vector<unsigned char> loadEngineFile(const std::string &file_name) {
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, std::ifstream::end);  // put the pointer at the end of the file
    auto length = (int) engine_file.tellg();  // the pos of the pointer is the length of the file
    engine_data.resize(length);
    engine_file.seekg(0, std::ifstream::beg);  // put the pointer at the beginning of the file
    // read 'length' chars to engine_data.data()
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
    return engine_data;
}

std::unordered_map<std::string, std::string> readFolderCategory(const std::string &cat_path) {
    std::unordered_map<std::string, std::string> folder_map;
    std::ifstream file(cat_path, std::ios::in);
    assert(file.is_open() && "Unable to load engine file.");
    std::string line;
    int cnt = 0;
    while (std::getline(file, line)) {
        folder_map.insert({line, std::to_string(cnt)});
        ++cnt;
    }
    return folder_map;
}

std::vector<std::string> getImgPathList(const std::string &input_path, bool sort = false, bool print = false) {
    std::vector<std::string> img_path_list;
    if (fs::is_directory(input_path)) {
        for (auto &p: fs::directory_iterator(input_path)) {
            img_path_list.push_back(p.path());
        }
    }
    if (sort) std::sort(img_path_list.begin(), img_path_list.end());  // alphabetic order
    // Print
    if (print) {
        std::cout << "Input images: (Should only contain image files)\n";
        for (auto &img_path: img_path_list) {
            std::cout << '\t' << img_path << '\n';
        }
        std::cout << '\n';
    }
    return img_path_list;
}

std::vector<std::pair<std::string, std::string>>
getImgPathAndCatList(const std::string &input_path, const std::string &cat_path, bool print = true, bool sort = false) {
    std::vector<std::pair<std::string, std::string>> img_path_cat_list;
    std::vector<std::string> img_folder_list;
    auto cat_map = readFolderCategory(cat_path);
    if (fs::is_directory(input_path)) {
        for (auto &p: fs::directory_iterator(input_path)) {
            if (fs::is_directory(p.path())) {
                img_folder_list.push_back(p.path());
            }
        }
        if (img_folder_list.empty()) {
            img_folder_list.push_back(input_path);
        }
        std::sort(img_folder_list.begin(), img_folder_list.end());
        for (auto &img_folder: img_folder_list) {
            auto last_slash = img_folder.find_last_of('/');
            auto folder_name = img_folder.substr(last_slash + 1, -1);
            auto img_path_list = getImgPathList(img_folder, sort, false);
//	        std::cout << "folder name: " << folder_name << "cat: " << cat_map[folder_name] << '\n';
            for (auto &img_path: img_path_list) {
                img_path_cat_list.emplace_back(img_path, cat_map[folder_name]);
            }
        }
    }
    if (print) {
        std::cout << "Input images and categories: (Should only contain image files)\n";
        for (auto &img_path_cat: img_path_cat_list) {
            auto img_path = img_path_cat.first;
            auto img_cat = img_path_cat.second;
            std::cout << "\t(path: \"" << img_path << "\", category: " << img_cat << ")\n";
        }
        std::cout << '\n';
    }
    return img_path_cat_list;
}

template<class T>
std::vector<std::pair<size_t, float>>
getTopCat(size_t k, T container) {
    std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, std::greater<>> q;
    for (int i = 0; i < container.size(); ++i) {
        if (q.size() < k) {
            q.push(std::pair<double, size_t>(container[i], i));
        } else if (q.top().first < container[i]) {
            q.pop();
            q.push(std::pair<double, size_t>(container[i], i));
        }
    }
    k = q.size();
    std::vector<std::pair<size_t, float>> k_pairs(k);
    for (int i = 0; i < k; ++i) {
        auto top = q.top();
        k_pairs[k - i - 1] = {top.second, top.first};
        q.pop();
    }
    return k_pairs;
}

template<class T>
std::vector<std::pair<size_t, float>>
printTopCat(size_t k, T container, const std::string &cat_path, const std::string &real_cat = "") {
    auto k_pairs = getTopCat(k, container);
    json cat = json::parse(std::ifstream{cat_path});
    std::unordered_map<std::string, std::string> label;
    if (!cat.at("id2label").is_null()) {
        cat.at("id2label").get_to(label);
    }
    std::cout << "[\n";
    for (int i = 0; i < k; ++i) {
        if (i) std::cout << ",\n";
        std::cout << "\t(id: " << k_pairs[i].first << ", "
                  << "category: \"" << label.at(std::to_string(k_pairs[i].first)) << "\", ";

        std::cout << "value: " << k_pairs[i].second << ")";
    }
    std::cout << "\n]";
    if (!real_cat.empty()) {
        std::cout << "\n(real_id: " << real_cat << ", "
                  << "real_category: \"" << label.at(real_cat) << ")";
    }
    return k_pairs;
}

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << '\n';
    }
} logger;


int main(int argc, char **argv) {

    // TODO: Read the JSON file path & custom tensor names
    std::string engine_file = "./trts/mobilenetv2.trt";
    std::string input_dir = "./val";
    std::string category_translation = "./config.json";
    std::string folder_category = "./imagenet_classes.txt";
    std::string preprocessor_file = "./preprocessor_config.json";
    std::string input_tensor_name = "input";
    std::string output_tensor_name = "output";
    std::string simple_cat = "-1";

#ifndef MY_DEBUG
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine_file> <input_directory> <input_tensor_name> <output_tensor_name> [preprocessor_json_file] [category_json_file] [expected_single_category]\n";
        return -1;
    }
    // argc >= 5
    engine_file = argv[1];
    input_dir = argv[2];
    input_tensor_name = argv[3];
    output_tensor_name = argv[4];
    if (argc > 5) {
        preprocessor_file = argv[5];
        if (argc > 6) {
            category_translation = argv[6];
            if (argc > 7) {
                simple_cat = argv[7];
            }
        }
    }
#endif
    // Print info
    std::cout << "Using engine file: " << engine_file << '\n'
              << "Using input directory: " << input_dir << '\n'
              << "Using categories from: " << category_translation << '\n'
              << "Using preprocessor from: " << preprocessor_file << '\n'
              << "Input tensor name: " << input_tensor_name << '\n'
              << "Output tensor name: " << output_tensor_name << '\n';

    auto plan = loadEngineFile(engine_file);

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "Runtime creation failed!\n";
        return -1;
    }
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine) {
        std::cerr << "Engine deserialization failed!\n";
        return -1;
    }
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Context creation failed!\n";
        return -1;
    }

    // Preprocessor config
    struct {  // resample type is not considered
        int resize_height = DEBUG_INPUT_H;
        int resize_width = DEBUG_INPUT_W;
        double rescale_factor = 1.0 / 255;
        std::array<double, 3> norm_mean{0.5, 0.5, 0.5};
        std::array<double, 3> norm_std{0.5, 0.5, 0.5};
    } config;

    // Preprocessor JSON parsing
    json preprocessor = json::parse(std::ifstream{preprocessor_file});
    std::unordered_set<std::string> keys;  // alias
    if (!preprocessor.at("_valid_processor_keys").is_null()) {
        preprocessor.at("_valid_processor_keys").get_to(keys);
    }
    if (keys.count("do_resize")) {
        preprocessor.at("size").at("height").get_to(config.resize_height);
        preprocessor.at("size").at("width").get_to(config.resize_width);
    }
    if (keys.count("do_rescale")) {
        preprocessor.at("rescale_factor").get_to(config.rescale_factor);
    }
    if (keys.count("do_normalize")) {
        preprocessor.at("image_mean").get_to(config.norm_mean);
        preprocessor.at("image_std").get_to(config.norm_std);
    }

    // Aliases for long func names
    auto now = std::chrono::high_resolution_clock::now;
    auto dur = [](std::chrono::time_point<std::chrono::system_clock> b,
                  std::chrono::time_point<std::chrono::system_clock> e) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(e - b).count();
    };

    // Input image list processing
    size_t img_cnt = 0;
    auto img_path_cat_list = getImgPathAndCatList(input_dir, folder_category, false);
    size_t total = img_path_cat_list.size();

//    for (auto& item : img_path_cat_list) {
//        std::cout << item.first << ' ' << item.second << '\n';
//    }
//    return 0;

    // Statistics
    std::vector<size_t> inference_time;
    size_t inference_acc_cnt = 0;

    for (const auto &img_path_cat: img_path_cat_list) {
        auto img_path = img_path_cat.first;
        auto img_cat = img_path_cat.second;
        if (simple_cat != "-1") {
            img_cat = simple_cat;
        }
        // Infer one image
        ++img_cnt;
        auto tick_preprocess = now();
        auto img = cv::imread(img_path);

        // Manual preprocessing: resize->rescale->normalize
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::resize(img, img, cv::Size{
                config.resize_width,
                config.resize_height
        });
        img.convertTo(img, CV_32FC3);  // CV_8UC3 -> CV_32FC3
        img = img * config.rescale_factor;  // [0, 255] -> [0.0, 1.0]
        // normalization
        img = img - cv::Scalar(config.norm_mean[0],
                               config.norm_mean[1],
                               config.norm_mean[2]
        );
        img = img / cv::Scalar(config.norm_std[0],
                               config.norm_std[1],
                               config.norm_std[2]
        );
        std::array<cv::Mat, 3> img_ch;
        cv::split(img, img_ch);
        std::vector<std::array<cv::Mat, 3>> batch;
        batch.push_back(img_ch);  // BS * C * H * W
        auto plane_size = config.resize_width * config.resize_height;

        // For Cuda mem copy
        auto data_ptr = std::malloc(DEBUG_BATCH_SIZE * 3 * plane_size * sizeof(float));
        for (int i = 0; i < 3; ++i) {
            cv::Mat mat_plane = batch[0][i];
            memcpy((float *) (data_ptr) + i * plane_size, mat_plane.data, plane_size * sizeof(float));
        }

        auto tock_preprocess = now();
        auto elapsed = dur(tick_preprocess, tock_preprocess);
        inference_time.push_back(elapsed);
//        auto fps_str = std::to_string(1000 / elapsed) + " fps";
//        std::cout << "Preprocessing time (" << img_cnt << "/" << total << "): " << std::to_string(elapsed) << " ms; ";

        auto tick_infer = now();

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        void *i_buffer, *o_buffer;
        size_t i_size = DEBUG_BATCH_SIZE * 3 * DEBUG_INPUT_H * DEBUG_INPUT_W * sizeof(float);
        size_t o_size = DEBUG_BATCH_SIZE * DEBUG_N_CAT * sizeof(float);
        cudaMalloc(&i_buffer, i_size);
        cudaMalloc(&o_buffer, o_size);

        // Copy an image to i_buffer
        cudaMemcpyAsync(i_buffer, data_ptr, i_size, cudaMemcpyHostToDevice, stream);

        context->setTensorAddress(input_tensor_name.data(), i_buffer);
        context->setTensorAddress(output_tensor_name.data(), o_buffer);

        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        // Copy results from o_buffer
        std::array<std::array<float, DEBUG_N_CAT>, DEBUG_BATCH_SIZE> result{};
        cudaMemcpyAsync(&result, o_buffer, o_size, cudaMemcpyDeviceToHost, stream);
        cudaStreamDestroy(stream);
        std::free(data_ptr);  // remember to free the custom ptr!

        auto tock_infer = now();
        elapsed = dur(tick_preprocess, tock_preprocess);
//        fps_str = std::to_string(1000 / elapsed) + " fps";
//        std::cout << "Inference time (" << img_cnt << "/" << total << "): " << std::to_string(elapsed) << " ms\n";

        // Output process
//        std::cout << img_path << '\n';
        size_t result_cnt = 0;
        for (auto &line: result) {
//            if (result_cnt++) std::cout << ",";
//            auto top_cat = printTopCat(1, line, category_translation, img_cat);
            auto top_cat = getTopCat(1, line);
            if (std::to_string(top_cat[0].first) == img_cat) {
                ++inference_acc_cnt;
            }
//            std::cout << '\n';
        }
    }

    // Average result summary
    auto time_sum = std::accumulate(inference_time.begin(), inference_time.end(), 0ul);
    std::cout
            << "\n==Summary==\n"
            << "Average inference time: " << std::to_string(time_sum / inference_time.size()) << " ms\n"
            << "Average FPS: " << std::to_string(1000.0 * double(inference_time.size()) / double(time_sum)) << '\n';
    std::cout << "Average accuracy: " << std::to_string(double(inference_acc_cnt) / double(total)) << '\n';


    return 0;
}

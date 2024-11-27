#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

#include <opencv2/opencv.hpp>

#include <vector>
#include <memory>

cv::Mat test_match(std::string& train_path, std::string& query_path){
    cv::Mat train_img = cv::imread(train_path);
    cv::Mat query_img = cv::imread(query_path);
    
    std::vector<cv::KeyPoint> train_kpts;
    std::vector<cv::KeyPoint> query_kpts;
    cv::Mat train_desc;
    cv::Mat query_desc;
    
    cv::Ptr<cv::SIFT> orb = cv::SIFT::create();
    orb->detectAndCompute(train_img, cv::noArray(), train_kpts, train_desc);
    orb->detectAndCompute(query_img, cv::noArray(), query_kpts, query_desc);
    
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    std::vector<std::vector<cv::DMatch>> nn_matches;
    // std::cout << "match suru zo ~" << std::endl;
    matcher->knnMatch(train_desc, query_desc, nn_matches, 2); // rows : num of train descriptor, cols : 1st, 2nd nearest
    
    std::vector<cv::KeyPoint> train_matched_kpts;
    std::vector<cv::KeyPoint> query_matched_kpts;
    std::vector<cv::DMatch> good_match;
    for(size_t i = 0; i < nn_matches.size(); ++i){
        cv::DMatch first_nn = nn_matches[i][0];
        cv::DMatch second_nn = nn_matches[i][1];
        if(first_nn.distance < second_nn.distance * 0.8){
            train_matched_kpts.push_back(train_kpts[first_nn.trainIdx]);
            query_matched_kpts.push_back(query_kpts[first_nn.queryIdx]);
            good_match.push_back(first_nn);
        }
    }
    
    std::cout << "==========================" << std::endl;
    std::cout << "train kpts : " << train_kpts.size() << std::endl;
    std::cout << "train matched kpts : " << train_matched_kpts.size() << std::endl;
    std::cout << "matched / all => " << static_cast<float>(train_matched_kpts.size()) / static_cast<float>(train_kpts.size()) << std::endl; 
    std::cout << "==========================" << std::endl;

    cv::Mat ret;
    cv::drawMatches(train_img, train_kpts, query_img, query_kpts, good_match, ret);
    // std::cout << "uo" << std::endl;
    // return std::move(ret);
    return ret;
}

int main(void){
    std::string train_path = "/home/toyozoshimada/Downloads/quad/cup_images/left0050.jpg";
    std::string query_1_path = "/home/toyozoshimada/Downloads/quad/cup_images/left0200.jpg";
    std::string query_2_path = "/home/toyozoshimada/Downloads/inu.jpg";

    cv::Mat train_to_q1 = test_match(train_path, query_1_path);
    // std::cout << "uouo" << std::endl;
    cv::Mat train_to_q2 = test_match(train_path, query_2_path);

    cv::imshow("train_to_q1", train_to_q1);
    cv::imshow("train_to_q2", train_to_q2);
    cv::waitKey(0);
    return 0;
}
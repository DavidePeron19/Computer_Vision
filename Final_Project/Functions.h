
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include "opencv2/highgui/highgui.hpp"
//#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>


enum class Flags {
    SIFT = true,
    ORB = false,
    BASE = true,
    TRANSFORMED = false,
    BFM = true,
    TEMPLATE = false,  
    AUTOMATIC = true,
    MANUAL = false
};

using namespace std;
using namespace cv;

void clear_terminal();

bool ask_binary (string, string);

int ask_octave_layers();

double ask_distance_ratio();

tuple<double, double> ask_multiple_threshold();

string ask_images_path();

vector<Mat>  loadImages(String, String, String);

void save_Results(Mat, string, string);

tuple<Mat, vector<KeyPoint>> Detect_Compute (Mat , Flags , int);

vector<DMatch> find_BFM_matches (Mat, Mat, Flags, double);

Mat Template_matching(Mat, Mat patch);

Mat Best_Flipping ( Mat, Mat, Flags, int, double);

Mat find_homography_CV (vector<KeyPoint> , vector<KeyPoint>, vector<DMatch> );

Mat find_affine_manual (vector<KeyPoint> , vector<KeyPoint> , vector<DMatch> );

Mat merge_filter (Mat , Mat);

void image_differences(Mat, Mat, string, string, bool);

vector<vector<Mat>> multiple_image_match (vector<Mat>, vector<Mat>, Flags, int, double, double, double);
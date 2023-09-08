#include "Functions.h"

using namespace std;
using namespace cv;



//function used to clear the terminal
//I have done a function since i don't know which platform the code will be run
//this takes into account windows linux and mac operating systems
void clear_terminal() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}


/*****************************************************************
*                           INPUT FUNCTIONS                      *
*      the following functions are used to interact with the     *
*  user and ask the input parameters needed for running the code *
 ****************************************************************/


/*
    used to ask the user a 2 possibility question return true if the
    first possibility has been chosen false if the second one is chosen
*/
bool ask_binary (string possibility1, string possibility2){
    cout << "1) " << possibility1  <<endl;
    cout << "2) " << possibility2  <<endl<<endl;

    // initialization
    bool valid = false;
    bool choice;
    string input;

    // the loop continues until a valid choice is made
    while(!valid) {
        cout << "Please enter either 1 or 2 then press enter:";
        // casting input to int to avoid errors 
        cin >> input;
        //checking if the input is a valid choice
        if(input=="1"){
            cout <<"\033[1;32mYour choice [1] was correctly registered\033[0m"<<endl<<endl;
            choice = true;
            valid = true;
        }
        else if (input=="2")
        {
            cout <<"\033[1;32mYour choice [2] was correctly registered\033[0m"<<endl<<endl;
            choice = false;
            valid = true;
        }
        else{
            cout << "\033[0;31mInvalid Choice\033[0m"<<endl;
        }
        
    }
    // Return the selected choice   
    return choice;
}


/*
    function used to ask the user the number of octave Layers for theSIFT algorithm
*/
int ask_octave_layers(){

    string input;
    int octaveLayers;
    bool valid = false;
    cout<<endl<<"\033[1;33mHow many Octave Layers do you want SIFT to use? [the higher the value the higher the computational time]\033[0m"<<endl;
    while(!valid){
        cout<<"Octave Layers [int>0]: ";
        cin >> input;
        
        // Create a stringstream from the input
        stringstream ss(input);
        
        // Read the value from the stringstream
        ss >> octaveLayers;
        
        // Check if the stringstream extraction was successful
        if (ss.fail()) {
            // Extraction failed
            cout<<"\033[0;31mThe value is not valid\033[0m"<<endl;
        }
        else{
            if(octaveLayers < 1){
                cout<<"\033[0;31mThe value is not valid\033[0m"<<endl;
            }
            else{
               cout <<"\033[1;32mYour choice: " +to_string(octaveLayers)+" was correctly registered\033[0m"<<endl<<endl;
                valid = true; 
            }
            
        }
    }    
    // Return the integer value
    return octaveLayers;
}


/*
    function used to ask the user distance ratio used to refine the matches
*/
double ask_distance_ratio(){

    string input;
    double distance_ratio;
    bool valid = false;
    valid = false;

    cout<<"\033[1;33mWhat ratio do you want to use to refine the matches?\033[0m"<<endl;
    while(!valid){        
        cout<<"Distance Ratio [double>1]: ";
        cin >> input;
        
        // Create a stringstream from the input
        stringstream ss(input);
        
        // Read the value from the stringstream
        ss >> distance_ratio;
        
        // Check if the stringstream extraction was successful
        if (ss.fail()) {
            // Extraction failed
            cout<<"\033[0;31mThe value is not valid\033[0m"<<endl;
        }
        else{
            //check if the value is acceptable
            if(distance_ratio<1){
                cout<<"\033[0;31mThe value is not valid\033[0m"<<endl;
            }
            else{
                cout <<"\033[1;32mYour choice: " +to_string(distance_ratio)+" was correctly registered\033[0m"<<endl<<endl;
                valid = true;
            }            
        }
    }    
    // Return the integer value
    return distance_ratio;
}


/*
    function used to ask the user the desired threshold when running in multiple image mode
    return a tuple of double containing < matches_threshold, sparsity_threshold >
*/
tuple<double, double> ask_multiple_threshold(){
    string input;
    double matches_threshold;
    double sparsity_threshold;

    bool valid = false;
    //ask for first input
    cout<<"\033[1;33mWhat match's distance do you want use as threshold to consider the match sparse?\033[0m"<<endl;
    while(!valid){
        cout<<"Match Distance Threshold [double>0]: ";
        cin >> input;
        
        // Create a stringstream from the input
        stringstream ss(input);
        
        // Read the value from the stringstream
        ss >> matches_threshold;
        
        // Check if the stringstream extraction was successful
        if (ss.fail()) {
            // Extraction failed
            cout<<"\033[0;31mThe value is not valid\033[0m"<<endl;
        }
        else{
            if(matches_threshold<1){
                cout<<"\033[0;31mThe value is not valid\033[0m"<<endl;
            }
            else{
                cout <<"\033[1;32mYour choice: " +to_string(matches_threshold)+" was correctly registered\033[0m"<<endl<<endl;
                valid = true;
            }   
        }
    } 
    //resetting the flag for the second value
    valid = false;
    //ask for second input
    cout<<"\033[1;33mWhat ratio between the number of sparse matches over the total number of matches you want to use as threshold to make a patch be associated to an image?\033[0m"<<endl;
    while(!valid){
        cout<<"Sparsity Ratio Threshold, must be in the interval [0;1]: ";
        cin >> input;
        
        // Create a stringstream from the input
        stringstream ss(input);
        
        // Read the value from the stringstream
        ss >> sparsity_threshold;
        
        // Check if the stringstream extraction was successful
        if (ss.fail()) {
            // Extraction failed
            cout<<"\033[0;31mThe value is not valid\033[0m"<<endl;
        }
        else{
            //check if the value is acceptable
            if(sparsity_threshold>1 || sparsity_threshold<0 ){
                cout<<"\033[0;31mThe value is not valid\033[0m"<<endl;
            }
            else{
                cout <<"\033[1;32mYour choice: " +to_string(sparsity_threshold)+" was correctly registered\033[0m"<<endl<<endl;
                valid = true;
            }            
        }
    } 

    // Return the tuple containing the values
    return make_tuple(matches_threshold,  sparsity_threshold);
}


/*
    function used to ask the user the path of the directory where the imnages
    are saved, it also checks that the direcotry exists
*/
string ask_images_path(){
    
    string path;
    bool valid = false;
    cout<<"\033[1;33mPlease enter the path of the directory from where you want to load the images?\033[0m"<<endl;    

    while(!valid){
        
        cout<<"Directory Path: ";
        cin >> path;

        if (!filesystem::is_directory(path)) {
            cout<<"\033[0;31mThe path provided is not valid directory\033[0m"<<endl;
        }
        else{
            cout <<"\033[1;32mPath correctly registered: \033[0m"+ path<<endl<<endl;
            valid = true;
        }         
    }

    return path;
} 


/*****************************************************************
*                           LOAD IMAGES                          *
*  Funcion to load pathces using directory path and to save them *
*   in a <Mat> vectorit takes for guaranteed that the images     *   
*               are jpg and we have 4 patches                    *
*****************************************************************/

vector<Mat> loadImages(string dir, string images_name, string title) {
    
    // Initialize vector of patches
    vector<Mat> images;

    int i=0;
    while (true){
        // Construct patch file path with extension        
        string path = dir + images_name + to_string(i) + ".jpg";
        //if i don't find the required image break loop 
        if (!ifstream(path)) {
                break;
        }
        // Read patch file
        Mat image = imread(path, IMREAD_COLOR);
        // Add patch to vector
        images.push_back(image);
        //updating counter
        i++;
    }   

    // Display number of patches read
    cout << title << images.size() << endl;

    return images;
}


/*****************************************************************
*                           SAVE RESULTS                         *
*       Funcion to save the results in a desired directory       *
*****************************************************************/

void save_Results(Mat image, string path, string file_name){

    //create the required directory if it doesn't exist
    filesystem::create_directories(path);

    //define full path of the image with file name
    string full_path =path+"/"+file_name+".jpg";

    //same the image
    imwrite(full_path, image);

}


/**********************************************************
*                   FEATURE DESCRIPTOR                    *
*   Function used to detect and compute the feature       *
*   in an image, it can handle both SIFT and ORB          *
**********************************************************/

tuple<Mat, vector<KeyPoint>> Detect_Compute (Mat image, Flags descriptor_type, int octave_layers){

    //initializing output variables
    Mat descriptors;
    vector<KeyPoint> keypoints;

    if(static_cast<bool>(descriptor_type)){
        //creating SIFT descriptor with no limits on the matches number and the desired octave layers
        Ptr<SIFT> feature_descriptor = SIFT::create(0, octave_layers);
        //detecting and computing descriptors and keypoints
        feature_descriptor->detectAndCompute(image, noArray(), keypoints, descriptors);
 
    }
    else{
        //creating ORB descriptor with no limits on the matches number
        Ptr<ORB> feature_descriptor = ORB::create(1000000);
        //detecting and computing descriptors and keypoints
        feature_descriptor->detectAndCompute(image, noArray(), keypoints, descriptors);
    }

    return  make_tuple(descriptors,  keypoints); 
}

/**********************************************************
*                   BFM MATCHES                           *
*   the following function will be used to find matches   *
*                   for SIFT or ORB                       *
**********************************************************/

vector<DMatch> find_BFM_matches (Mat descriptors_image, Mat descriptors_patch, Flags descriptor_type, double distance_ratio){
    
    //output vector
    vector<DMatch> matches;        

    // definition of the matcher
    BFMatcher matcher(NORM_L2);
    if(static_cast<bool>(descriptor_type)){
        //using NORM_L2 in case of SIFT
        BFMatcher matcher(NORM_L2);
    }
    else{
        //using NORM_HEMMING in case of ORB
        BFMatcher matcher(NORM_HAMMING);
    }   


    //support vectors for first filtering
    vector<DMatch> matches_filter_1;
    vector<vector<DMatch>> NORM_matches;
    matcher.knnMatch(descriptors_patch, descriptors_image, NORM_matches, 2);

    // Filter the matches based on distance ratio
    for (size_t j = 0; j < NORM_matches.size(); ++j) {
        if (NORM_matches[j][0].distance < 0.8 * NORM_matches[j][1].distance) {
            matches_filter_1.push_back(NORM_matches[j][0]); //add element at the end of the vector
        }            
    }

    // Filter the matches based on distance between one another
    // used to refine even more the matches
    float min_distance = FLT_MAX;

    //fining minimum distance
    for (size_t j = 0; j < matches_filter_1.size(); j++) {
        if (matches_filter_1[j].distance < min_distance) {
            min_distance = matches_filter_1[j].distance;
        }
    }

    //keeping only matches that are closer
    for (size_t j = 0; j < matches_filter_1.size(); j++) {
        if (matches_filter_1[j].distance <= distance_ratio * min_distance) {
            matches.push_back(matches_filter_1[j]);
        }
    }      
    return matches;
}


/**********************************************************
*                  TEMPLATE MATCHING                      *
*   the following function will be used to do template    *
*   matching it takes as input the corrupted image and a  *
*   patch and returns the image with the patch inserted   *
*   HEAVILY INSPIRED BY OPENCV DOCUMENTATION              *
**********************************************************/
Mat Template_matching(Mat image_to_complete, Mat patch){

    //creating a copy of the image to patch
    Mat image_out=image_to_complete.clone();

    //creating the map of comparisons as described in the documentation
    Mat result;
    int result_cols = image_out.cols - patch.cols + 1;
    int result_rows = image_out.rows - patch.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    //running the template matching function
    matchTemplate(image_out, patch, result, TM_CCOEFF_NORMED);
    double min_value, max_value;
    Point min_location, max_location;

    //passing values by reference as required
    minMaxLoc(result, &min_value, &max_value, &min_location, &max_location, Mat());

    //creating a mask that represents the region in image_out where patch will be attached
    Rect position_mask(max_location, Size(patch.cols, patch.rows));
    //attaching the patch to the image in the position required by the mask
    patch.copyTo(image_out(position_mask));

    return image_out;
}









/**********************************************************
*                    BEST FLIPPING                        *
*  the following function find which is the best flipping *
*  of the patch given as input and return the patch       *
*                  flipped accordingly                    *
**********************************************************/

Mat Best_Flipping ( Mat patch, Mat descriptors_image, Flags descriptor_type, int octave_layers, double distance_ratio ){
    
    //starts from -1 to be sure that at least one we change the output image
    int best_num_matches = -1;
    Mat image_out;

    //for loop starts from -1 to use all the possible options of flipping
    for (int i=-1; i<=2; i++){
        
        //define necessary structures
        vector<KeyPoint> kp_patch; 
        Mat desc_patch;
        vector<DMatch> matches;
        Mat flipped_patch;

        //flip the image
        //first 3 time try with different flipping (horizontal, verical, diagonal)
        //4-th timewith original image
        if (i<=1){
            flip(patch, flipped_patch, i);
        }
        else{
            flipped_patch = patch.clone();
        }

        //find features and matches of the flipped patch
        tie (desc_patch, kp_patch) = Detect_Compute(flipped_patch, descriptor_type, octave_layers);

        matches = find_BFM_matches(descriptors_image, desc_patch, descriptor_type, distance_ratio);

        //updating output matrix if i obtain better results
        if(int(matches.size())>best_num_matches){
            best_num_matches = int(matches.size());
            image_out = flipped_patch.clone();
        }
    }
    return image_out;
}


/**********************************************************
*                    HOMOGRAPHY                           *
*  the following functions will find the homography       *
*  matrix associated with the image and patch             *
*                  flipped accordingly                    *
**********************************************************/

/*
    function to find homography matrix by using library functions
    the output will be a Mat CV_32F to be compatible with WarpPerspective
*/

Mat find_homography_CV (vector<KeyPoint> KP_image, vector<KeyPoint> KP_patch, vector<DMatch> matches){

    //define homography matrix as an empty matrix
    Mat H =cv::Mat::zeros(3, 3, CV_32F);

    //find the points in the image and patch that matches
    vector<Point2f> patchPoints, imagePoints;
    for (int i = 0; i < matches.size(); i++) {
        imagePoints.push_back(KP_image[matches[i].trainIdx].pt);
        patchPoints.push_back(KP_patch[matches[i].queryIdx].pt);
    }

    //using library function to find the homography matrix
    H = findHomography(patchPoints, imagePoints, RANSAC); 

    //converting the matrix into the required standard
    H.convertTo(H, CV_32F);

    return H;
}


/*
    the function will manually find the best affine transformation
    and return the correspondent affine transformation matrix
*/

Mat find_affine_manual (vector<KeyPoint> KP_image, vector<KeyPoint> KP_patch, vector<DMatch> matches){

    //define affine transformation matrix matrix as an empty matrix
    Mat affine_out(2, 3, CV_32F, Scalar(0));

    //find the points in the image and patch that matches
    vector<Point2f> patchPoints, imagePoints;
    for (int i = 0; i < matches.size(); ++i) {
        imagePoints.push_back(KP_image[matches[i].trainIdx].pt);
        patchPoints.push_back(KP_patch[matches[i].queryIdx].pt);
    }

    //define  performance parametes
    int maxIters = 100;
    int dist_threshold = 3;

    //initialization
    int best_correspondencies = 0;    

    // iteratively find the best affine transformation for the patch
    for (int i = 0; i < maxIters ; i++) {
        
        vector<Point2f> patchsubset(3);
        vector<Point2f> imagesubset(3);

        //find 3 random points for iteration to obtain the affine transformation
        for (int j = 0; j < 3; j++) {
            int idx = rand()%(patchPoints.size() - 1);
            patchsubset[j] = patchPoints[idx];
            imagesubset[j] = imagePoints[idx];
        }

        //assigning values for matrices A, b as in the slides
        Mat A(6, 6, CV_32F, Scalar(0));
        Mat b(6, 1, CV_32F, Scalar(0));
        for (int j=0; j<3; j++){
            int idx = 2*j;
            A.at<float>(idx, 0) = patchsubset[j].x;
            A.at<float>(idx, 1) = patchsubset[j].y;
            A.at<float>(idx, 4) = 1;
            A.at<float>(idx+1, 2) = patchsubset[j].x;
            A.at<float>(idx+1, 3) = patchsubset[j].y;
            A.at<float>(idx+1, 5) = 1;
            b.at<float>(idx) = imagesubset[j].x;
            b.at<float>(idx+1) = imagesubset[j].y;
        }

        //solving equation with least squares
        Mat solution;
        solve(A, b, solution, DECOMP_SVD);

        Mat affine_temp(2, 3, CV_32F, Scalar(0));
        affine_temp.at<float>(0,0) = solution.at<float>(0);
        affine_temp.at<float>(0,1) = solution.at<float>(1);
        affine_temp.at<float>(0,2) = solution.at<float>(4);

        affine_temp.at<float>(1,0) = solution.at<float>(2);
        affine_temp.at<float>(1,1) = solution.at<float>(3);
        affine_temp.at<float>(1,2) = solution.at<float>(5);
       
        //find how many correspondencies does the affine transformation get
        int correspondencies = 0;
        for (int j = 0; j < patchPoints.size(); j++) {

            Point2f projected_pt;
            Mat point_mat = (Mat_<float>(3, 1) << patchPoints[j].x, patchPoints[j].y, 1.0);         
            Mat result_mat = affine_temp * point_mat;
            projected_pt.x = result_mat.at<float>(0, 0);
            projected_pt.y = result_mat.at<float>(1, 0);
            //calculating distance between projected point and real point
            double distance = norm(projected_pt - imagePoints[j]);

            if (distance < dist_threshold) {
                correspondencies++;
            }   
        }
        //updating A if i get more correspondencies than the previous best
        if (correspondencies > best_correspondencies) {
            best_correspondencies = correspondencies;
            affine_out = affine_temp;
        }
    }
    return affine_out;

}


/**********************************************************
*              MERGE AND FILTER                           *
**********************************************************/
Mat merge_filter (Mat image, Mat patch){

    //taking only point of the patch that are not black
    Mat mask = patch > 0;

    // Convert the mask to grayscale
    Mat mask_float;
    mask.convertTo(mask_float, CV_32FC3, 1.0 / 255.0);

    Mat eroded_mask;
    int erosion_size = 2; 
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
    //removes possible noise in the mask and erodes it
    morphologyEx(mask_float, eroded_mask, MORPH_CLOSE,  element);
    //removes the black border that will form by making the mask rectangular and smaller than the original patch
    morphologyEx(eroded_mask, eroded_mask, MORPH_RECT,  element);

    // Ensure the mask is of type CV_32FC3
    if (eroded_mask.type() != CV_32FC3) {
        eroded_mask.convertTo(eroded_mask, CV_32FC3);
    }

    // Resize the mask to match the size of the images
    resize(eroded_mask, eroded_mask, image.size(), 0, 0, INTER_CUBIC);

    cvtColor(eroded_mask, eroded_mask, COLOR_BGR2GRAY);

    vector<Mat> image_ch, patch_ch;
    split(image, image_ch);
    split(patch, patch_ch);

    // Apply alpha blending using the feathered mask for each channel
    vector<Mat> blended_ch(3);

    for (int j = 0; j < 3; ++j) {

        Mat image_patch_float, image_float;
        patch_ch[j].convertTo(image_patch_float, CV_32FC1);
        image_ch[j].convertTo(image_float, CV_32FC1);

        //applying the mask created previously
        blended_ch[j] = image_float.mul(1.0 - eroded_mask) + image_patch_float.mul(eroded_mask);
        blended_ch[j].convertTo(blended_ch[j], CV_8UC1);
    }

    // Merge the channels back into the final image
    merge(blended_ch, image);

    return image;
}



/**********************************************************
*              IMAGE DIFFERENCES                         *
*   this function computes the difference between the     *
*   image in input displays and saves them                *
**********************************************************/

void image_differences(Mat original_image, Mat modified_image, string path, string file_name, bool save_results){

    // compute the difference between the images
    Mat differences;
    absdiff(original_image, modified_image, differences);

    // convert to grayscale
    cvtColor(differences, differences, cv::COLOR_BGR2GRAY);

    // highlighting only the major differences and 
    // convertig into a black and whtie image
    threshold(differences, differences, 30, 255, cv::THRESH_BINARY);

    //saving only if required
    if(save_results){
        save_Results(differences, path, file_name);
    }    

    //showing the differences image
    namedWindow(file_name, WINDOW_NORMAL);
    imshow(file_name, differences);
    waitKey(0);


}




/**********************************************************
*               MULTIPLE IMAGES MATCH                     *
*     this function associate each patch to an image      *
*     and returns a vector  op vector containing the      *
*           patches relative to an image                  *
**********************************************************/

vector<vector<Mat>> multiple_image_match (vector<Mat> images, vector<Mat>patches, Flags patches_type,  int octavelayers, double distance_ratio, double matches_distance_threshold, double sparsity_threshold){
    
    //defining output vector
    vector<vector<Mat>> out_patches;    

    //associating patches to each image
    for(int i=0; i<images.size(); i++){

        //since it's computationally expensive i write where we are
        cout<<"\033[1;31mWorking on image #"+to_string(i)+"\033[0m"<<endl;

        //computing keypoints and descrtiptors for the image
        vector<KeyPoint> kp_image;
        Mat desc_image;
        tie (desc_image, kp_image) = Detect_Compute(images[i], Flags::SIFT, octavelayers);

        //vector containin the good patches for the image
        vector<Mat> good_patches;

        for (int j = 0; j < patches.size(); j++)
        {
            //since it's computationally expensive i write where we are
            cout<<"Working on patch #"+to_string(j)<<endl;

            vector<KeyPoint> kp_patch; 
            Mat desc_patch;
            vector<DMatch> matches;

            //find best flipping of the image based on number of matches and change the new patch with the best one
            //this is done ony for transformed patches
            if(!static_cast<bool>(patches_type)){
                patches[j] = Best_Flipping (patches[j], desc_image, Flags::SIFT, octavelayers, distance_ratio);
            }

            // Detect and compute SIFT features of the pach        
            tie (desc_patch, kp_patch) = Detect_Compute(patches[j], Flags::SIFT, octavelayers);

            // adding descriptor and keypoint to the vectors
            matches = find_BFM_matches(desc_image, desc_patch, Flags::SIFT, distance_ratio);

            //checking how sparse are the matches 
            int sparseMatchCount = 0;
            for (const auto& match : matches) {
                if (match.distance > matches_distance_threshold) {                    
                    sparseMatchCount++;
                }
            }

            double sparsityRatio = static_cast<double>(sparseMatchCount) / static_cast<double>(matches.size());    

            //adding only if i get good results
            if(sparsityRatio<sparsity_threshold){
                good_patches.push_back(patches[j]);
            }
        }
        
        //adding good patches to che correspondent image
        out_patches.push_back(good_patches);
    }

    return out_patches;
}















#include "main_parts.h"

/**********************************************************
*              SINGLE IMAGE DESCRIPTORS                   *
*  the following function will manage the single images   *
*      part of the code except for template matching      *
**********************************************************/

void single_image_descriptors(string path, Flags patches_type, Flags descriptor_type, Flags homography_type, int octave_layers, double distance_ratio, bool draw_matches, bool save_results){
     
    //getting original images just for final difference check
    Mat original_image = imread(path + "/original_image.jpg");
    
    //getting corrupted image
    Mat image_in = imread(path + "/image_to_complete.jpg");
    namedWindow("Image to complete", WINDOW_NORMAL);
    imshow("Image to complete", image_in);
    waitKey(0);


    // detecting and computing descriptors and keypoints of the corrupted image       
    vector<KeyPoint> keypoints_image;
    Mat descriptors_image;
    
    tie (descriptors_image, keypoints_image) = Detect_Compute(image_in, descriptor_type, octave_layers);
    
    // getting the patches
    vector<Mat> patches;
    if(static_cast<bool>(patches_type)){
        patches = loadImages(path, "/patch_", "Base patches: ");
        //updating the path to eventually save the results
        path = path+"/results/base_pathces";
    }
    else{
        patches = loadImages(path, "/patch_t_", "Affine Tranformed patches: ");
        //updating the path to eventually save the results
        path = path+"/results/transformed_pathces";
    }
    //vectors containg KP and descriptors of the patches Matches
    vector<vector<KeyPoint>> keypoints_patches;
    vector<Mat> descriptors_patches;

    //vector containing the matches of every patch with the corrupted image
    vector<vector<DMatch>> matches;
    
    // Iterate through all the patch files and perform:
    // SIFT feature extraction
    // find best flipping of the image 
    // find matches
    for (int i = 0; i < patches.size(); i++) {

        vector<KeyPoint> kp_patch; 
        Mat desc_patch;

        //perform best flipping only with transformed patches to be less computationally heavy
        if(!static_cast<bool>(patches_type)){

            //find best flipping of the image based on number of matches and change the new patch with the best one
            patches[i] = Best_Flipping (patches[i], descriptors_image, descriptor_type, octave_layers, distance_ratio);
        }

        // Detect and compute SIFT features of the pach        
        tie (desc_patch, kp_patch) = Detect_Compute(patches[i], descriptor_type, octave_layers);

        // adding descriptor and keypoint to the vectors
        keypoints_patches.push_back(kp_patch);
        descriptors_patches.push_back(desc_patch); 
        
        // Match SIFT descriptors of the image and the patch
        // and adding it to the vector containing matches       
        matches.push_back(find_BFM_matches (descriptors_image, descriptors_patches[i], descriptor_type, distance_ratio)); 
    
        // generating the drawMatches only if necessary to be more efficient
        if(draw_matches | save_results){

            Mat good_matches;
            
            drawMatches(patches[i], keypoints_patches[i], image_in, keypoints_image, matches[i], good_matches);

            //saving the matches only if required  
            if(save_results){
                cout<<"Saving Good Matches "+to_string(i)<<endl;
                if(static_cast<bool>(descriptor_type)){
                    save_Results(good_matches, path, "SIFT_Good_Matches_" + to_string(i));
                }
                else{
                    save_Results(good_matches, path, "ORB_Good_Matches_" + to_string(i));
                }
            }
            //showing the matches only if required
            if(draw_matches){
                namedWindow("Good Matches " + to_string(i), WINDOW_NORMAL);
                imshow("Good Matches " + to_string(i), good_matches);
                waitKey(0);
            }
        }
    }

    //creating a copy of the original image to then merge with the patches
    Mat image_copy = image_in.clone();
    
    for (int i = 0; i < patches.size(); i++) {  

        //checking if we have enough matches to find a transformation and then merge the image
        if(matches[i].size()>=4){


            Mat warped_Patch;
            if  (static_cast<bool>(homography_type)) {
                Mat H =find_homography_CV (keypoints_image, keypoints_patches[i], matches[i]);
                //using warpPerspective since we have an homography
                warpPerspective(patches[i], warped_Patch, H, image_in.size());                    
            }
            else {
                Mat A = find_affine_manual (keypoints_image, keypoints_patches[i], matches[i]);                
                //using warpAffine since we have an affine transfomation    
                warpAffine(patches[i], warped_Patch, A, image_in.size());
            }
            
            image_copy = merge_filter (image_copy, warped_Patch);

        }
        else{
            cerr<<"\033[0;31mWe don't have enough matches for patch \033[0m"+ to_string(i)<<endl;
        }

        namedWindow("Blended Image with patch up to " + to_string(i), WINDOW_NORMAL);
        imshow("Blended Image with patch up to " + to_string(i), image_copy);

        //saving the results if required
        if(save_results){
            cout<<"Saving final results up to patch "+to_string(i)<<endl;
            if(static_cast<bool>(descriptor_type)){
                if(static_cast<bool>(homography_type)){
                    save_Results(image_copy, path, "SIFT_CV_Blended_up_to_patch_" + to_string(i));
                }
                else{
                    save_Results(image_copy, path, "SIFT_MANUAL_Blended_up_to_patch_" + to_string(i));
                }  
            }
            else{
                if(static_cast<bool>(homography_type)){
                    save_Results(image_copy, path, "ORB_CV_Blended_up_to_patch_" + to_string(i));
                }
                else{
                    save_Results(image_copy, path, "ORB_MANUAL_Blended_up_to_patch_" + to_string(i));
                }
            }
        }
        waitKey(0);
    }

    // showing differences between original image and the patched one and saving if required
    if(static_cast<bool>(descriptor_type)){
        if(static_cast<bool>(homography_type)){
            image_differences(original_image, image_copy, path, "differences_SIFT_CV", save_results);
        }
        else{
            image_differences(original_image, image_copy, path, "differences_SIFT_MANUAL", save_results);            
        }  
    }
    else{
        if(static_cast<bool>(homography_type)){
            image_differences(original_image, image_copy, path, "differences_ORB_CV", save_results);
        }
        else{
            image_differences(original_image, image_copy, path, "differences_ORB_MANUAL", save_results);            
        } 
    }
}


/**********************************************************
*                    MULTIPLE IMAGES                      *
*         the following function will manage the          *
*            multiple images part of the code             *
**********************************************************/

 void multiple_images(string path, Flags patches_type, Flags homography_type, int octave_layers, double distance_ratio, double matches_distance_threshold, double sparsity_threshold, bool draw_matches, bool save_results){

    //getting original images just for final difference check
    vector<Mat> original_images = loadImages(path, "/original_image_", "Original Images: ");

    //getting images
    vector<Mat> images = loadImages(path, "/image_to_complete_", "Images: ");

    // getting the patches
        vector<Mat> patches_in;
        if(static_cast<bool>(patches_type)){
            patches_in = loadImages(path, "/patch_", "Base patches: ");
            //updating the path to eventually save the results
            path = path+"/results/base_pathces";
        }
        else{
            patches_in = loadImages(path, "/patch_t_", "Affine Tranformed patches: ");
            //updating the path to eventually save the results
            path = path+"/results/transformed_pathces";
        }

    vector<vector<Mat>> ordered_patches = multiple_image_match (images, patches_in, patches_type, octave_layers, distance_ratio, matches_distance_threshold, sparsity_threshold);
    for(int t=0; t<images.size(); t++){

        //taking the t-th element of the 2 vectors and then running almost exaclty the same code as in single image
        Mat image_in =images[t];
        vector<Mat> patches = ordered_patches[t];


         // detecting and computing descriptors and keypoints of the corrupted image       
        vector<KeyPoint> keypoints_image;
        Mat descriptors_image;

        tie (descriptors_image, keypoints_image) = Detect_Compute(image_in, Flags::SIFT, octave_layers);
        
        //vectors containg KP and descriptors of the patches Matches
        vector<vector<KeyPoint>> keypoints_patches;
        vector<Mat> descriptors_patches;

        //vector containing the matches of every patch with the corrupted image
        vector<vector<DMatch>> matches;
        
        // Iterate through all the patch files and perform:
        // SIFT feature extraction
        // find best flipping of the image 
        // find matches
        for (int i = 0; i < patches.size(); i++) {

            vector<KeyPoint> kp_patch; 
            Mat desc_patch;

            //perform best flipping only with transformed patches to be less computationally heavy
            if(!static_cast<bool>(patches_type)){
                //find best flipping of the image based on number of matches and change the new patch with the best one
                patches[i] = Best_Flipping (patches[i], descriptors_image, Flags::SIFT, octave_layers, distance_ratio);
            }

            // Detect and compute SIFT features of the pach        
            tie (desc_patch, kp_patch) = Detect_Compute(patches[i], Flags::SIFT, octave_layers);

            // adding descriptor and keypoint to the vectors
            keypoints_patches.push_back(kp_patch);
            descriptors_patches.push_back(desc_patch); 
            
            // Match SIFT descriptors of the image and the patch
            // and adding it to the vector containing matches       
            matches.push_back(find_BFM_matches (descriptors_image, descriptors_patches[i], Flags::SIFT, distance_ratio)); 
        
            // generating the drawMatches only if necessary to be more efficient
            if(draw_matches | save_results){

                Mat good_matches;
                
                drawMatches(patches[i], keypoints_patches[i], image_in, keypoints_image, matches[i], good_matches);

                //saving the matches only if required  
                if(save_results){
                    cout<<"Saving Good Matches "+to_string(i)<<endl;
                        save_Results(good_matches, path, "MULTIPLE_SIFT_image_"+to_string(t)+"_Good_Matches_" + to_string(i));                
                }
                //showing the matches only if required
                if(draw_matches){
                    namedWindow("Good Matches " + to_string(i)+ " Image "+to_string(t), WINDOW_NORMAL);
                    imshow("Good Matches " + to_string(i)+ " Image "+to_string(t), good_matches);
                    waitKey(0);
                }
            }
        }

        //creating a copy of the original image to then merge with the patches
        Mat image_copy = image_in.clone();
        
        for (int i = 0; i < patches.size(); i++) {  
    
            //checking if we have enough matches to find a transformation and then merge the image
            if(matches[i].size()>=4){


                Mat warped_Patch;
                if  (static_cast<bool>(homography_type)) {
                    Mat H =find_homography_CV (keypoints_image, keypoints_patches[i], matches[i]);
                    //using warpPerspective since we have an homography
                    warpPerspective(patches[i], warped_Patch, H, image_in.size());                    
                }
                else {
                    Mat A = find_affine_manual (keypoints_image, keypoints_patches[i], matches[i]);                
                    //using warpAffine since we have an affine transfomation    
                    warpAffine(patches[i], warped_Patch, A, image_in.size());
                }
                
                image_copy = merge_filter (image_copy, warped_Patch);

            }
            else{
                cerr<<"\033[0;31mWe don't have enough matches for patch \033[0m"+ to_string(i)<<endl;
            }

            namedWindow("Blended Image "+to_string(t)+" with patch up to " + to_string(i), WINDOW_NORMAL);
            imshow("Blended Image "+to_string(t)+" with patch up to " + to_string(i), image_copy);
            waitKey(0);

            //saving the results if required
            if(save_results){
                cout<<"Saving final results up to patch "+to_string(i)<<endl;
                if(static_cast<bool>(homography_type)){
                    save_Results(image_copy, path, "MULTIPLE_SIFT_CV_image_"+to_string(t)+"_Blended_up_to_patch_" + to_string(i));
                }
                else{
                    save_Results(image_copy, path, "MULTIPLE_SIFT_MANUAL_image_"+to_string(t)+"_Blended_up_to_patch_"+ to_string(i));
                }  
            }
        }

        //showing differences between final image and original image        
        if(static_cast<bool>(homography_type)){
            image_differences(original_images[t], image_copy, path, "differences_MULTIPLE_SIFT_CV_image_"+to_string(t), save_results);
        }
        else{
            image_differences(original_images[t], image_copy, path, "differences_MULTIPLE_SIFT_MANUAL_image_"+to_string(t), save_results);
        }
    }
} 


/**********************************************************
*                 SINGLE IMAGE TEMPLATE                   *
*      the following function will manage the single      *
*      image with template matching  part of the code     *
**********************************************************/
void single_image_template (string path, bool save_results){

    //getting original images just for final difference check
    Mat original_image = imread(path + "/original_image.jpg");
    
    //getting corrupted image
    Mat image_in = imread(path + "/image_to_complete.jpg");
    namedWindow("Image to complete", WINDOW_NORMAL);
    imshow("Image to complete", image_in);
    waitKey(0);


    vector<Mat> patches = loadImages(path, "/patch_", "Base patches: ");
    //updating the path to eventually save the results
    path = path+"/results/base_pathces";

    //creating a copy of the original image to then merge with the patches
    Mat image_copy = image_in.clone();

    //running templte match to every patch
    for(int i=0; i<patches.size(); i++){

        //doing the function created for template matching
        image_copy = Template_matching(image_copy, patches[i]);

        //saving results if required
        if(save_results){
            cout<<"Saving final results up to patch "+to_string(i)<<endl;
            save_Results(image_copy, path, "TEMPLATE_Blended_up_to_patch_" + to_string(i));
        }

        //showing the progress
        namedWindow("Blended Image with patch up to " + to_string(i), WINDOW_NORMAL);
        imshow("Blended Image with patch up to " + to_string(i), image_copy);
        waitKey(0);
    }

    // showing differences between original image and the patched one and saving if required
    image_differences(original_image, image_copy, path, "differences_TEMPLATE", save_results);    
}







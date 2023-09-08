
#include <opencv2/opencv.hpp>
#include <iostream>
#include "main_parts.h"
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;


int main()
{   
    //disable all warnings that clog the terminal
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    //initialization of selectors
    bool  quantity_type, patches_type, tecnique_type, descriptor_type, homography_type;

    //initialization of the input parameters
    int octave_layers = 0; //used to decide the number of octave layers used in SIFT

    double distance_ratio; // used to refine the matches

    double match_distance_threshold; //used in multiple image to decide the distance to considera a match saprse or not

    double sparsity_ratio_threshold; //used in multiple images to set the threshold of the percentege of sparse matches

    string path; //contains the path of the directory from where we are going to read the images

    // initialization of other flags
    bool draw_matches;
    bool save_results;


    /*          DEBUG                    */
    //string path = "C:/Users/david/Desktop/cv/FinalProject/Dataset/venezia";
    //single_image_descriptors(path, Flags::TRANSFORMED, Flags::ORB, Flags::MANUAL, 5, 3, true, false);
    //multiple_images(path, Flags::BASE, Flags::MANUAL, 5, 3, 30, 0.3, false, true);

    cout<<"\033[1;33mWhat mode do you want to run? (folder needs to contain images in the required format, see report)\033[0m"<<endl;
    quantity_type = ask_binary("Single Corrupted Image", "Multiple Corrupted Image [uses only SIFT and more computationally expensive]");

//enters single image mode
    if(quantity_type){

        cout<<"\033[1;33mWhat type of patches do you want to use?\033[0m"<<endl;
        patches_type = ask_binary("Basic Patches", "Transformed Patches [uses only SIFT or ORB and more computationally expensive]");
    
    //enters base patches mode
        if(patches_type){
           
            cout<<"\033[1;33mWhat type of tecnique do you want to use to reconstruct the image??\033[0m"<<endl;
            tecnique_type = ask_binary("Feature Descriptor", "Template Matching");
        
        // entering feature descriptor mode   
            if(tecnique_type){
                
                cout<<"\033[1;33mWhat feature descriptor do you want to use? \033[0m"<<endl;
                descriptor_type = ask_binary("SIFT", "ORB");

            //entering SIFT mode
                if(descriptor_type){

                    cout<<"\033[1;33mHow do you want to calculate Homography matrix?\033[0m"<<endl;
                    homography_type = ask_binary("CV Library Function", "Manual RANSAC Function");

                //enters automatic mode
                    if(homography_type){

                        clear_terminal();
                        cout<<"\033[0;36mYou have selected: Basic Patches, Single Corrupted Image, Descriptor: SIFT, MAtcher: BFMatcher L2_NORM,  CV RANSAC\033[0m"<<endl<<endl;
                        
                        octave_layers = ask_octave_layers();
                        distance_ratio = ask_distance_ratio();
                        path = ask_images_path();
                        cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                        draw_matches = ask_binary("YES", "NO");
                        cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                        save_results = ask_binary("YES", "NO");

                        single_image_descriptors(path, Flags::BASE, Flags::SIFT, Flags::AUTOMATIC, octave_layers, distance_ratio, draw_matches, save_results);

                    }
                //enters manual mode
                    else{

                        clear_terminal();
                        cout<<"\033[0;36mYou have selected: Basic Patches, Single Corrupted Image, Descriptor: SIFT, MAtcher: BFMatcher L2_NORM,  Manual RANSAC\033[0m"<<endl<<endl;
                        
                        octave_layers = ask_octave_layers();
                        distance_ratio = ask_distance_ratio();
                        path = ask_images_path();
                        cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                        draw_matches = ask_binary("YES", "NO");
                        cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                        save_results = ask_binary("YES", "NO");

                        single_image_descriptors(path, Flags::BASE, Flags::SIFT, Flags::MANUAL, octave_layers, distance_ratio, draw_matches, save_results);
                  
                    }
                }
            //entering ORB mode
                else{

                    cout<<"\033[1;33mHow do you want to calculate Homography matrix?\033[0m"<<endl;
                    homography_type = ask_binary("CV Library Function", "Manual RANSAC Function");

                //enters automatic mode
                    if(homography_type){

                        clear_terminal();
                        cout<<"\033[0;36mYou have selected: Basic Patches, Single Corrupted Image, Descriptor: ORB, MAtcher: BFMatcher HEMMING_NORM,  CV RANSAC\033[0m"<<endl<<endl;
                        
                        distance_ratio = ask_distance_ratio();
                        path = ask_images_path();
                        cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                        draw_matches = ask_binary("YES", "NO");
                        cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                        save_results = ask_binary("YES", "NO");

                        single_image_descriptors(path, Flags::BASE, Flags::ORB, Flags::AUTOMATIC, octave_layers, distance_ratio, draw_matches, save_results);

                    }
                //enters manual mode
                    else{

                        clear_terminal();
                        cout<<"\033[0;36mYou have selected: Basic Patches, Single Corrupted Image, Descriptor: ORB, MAtcher: BFMatcher HEMMING_NORM,  Manual RANSAC\033[0m"<<endl<<endl;
                        
                        distance_ratio = ask_distance_ratio();
                        path = ask_images_path();
                        cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                        draw_matches = ask_binary("YES", "NO");
                        cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                        save_results = ask_binary("YES", "NO");

                        single_image_descriptors(path, Flags::BASE, Flags::ORB, Flags::MANUAL, octave_layers, distance_ratio, draw_matches, save_results);
                  
                    }
                }
            }
        //entering template matching mode
            else{

                clear_terminal();
                cout<<"\033[0;36mYou have selected: Basic Patches, Single Corrupted Image, Template Matching\033[0m"<<endl<<endl;

                path = ask_images_path();
                cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                save_results = ask_binary("YES", "NO");
                single_image_template(path, save_results);
                        

            }
        }
    //enters transformed patches mode 
        else{

            cout<<"\033[1;33mWhat feature descriptor do you want to use? \033[0m"<<endl;
            descriptor_type = ask_binary("SIFT", "ORB");

        //entering SIFT mode
            if(descriptor_type){

                cout<<"\033[1;33mHow do you want to calculate Homography matrix?\033[0m"<<endl;
                homography_type = ask_binary("CV Library Function", "Manual RANSAC Function");

            //enters automatic mode
                if(homography_type){

                    clear_terminal();
                    cout<<"\033[0;36mYou have selected: Transformed Patches, Single Corrupted Image, Descriptor: SIFT, MAtcher: BFMatcher L2_NORM,  CV RANSAC\033[0m"<<endl<<endl;
                    
                    octave_layers = ask_octave_layers();
                    distance_ratio = ask_distance_ratio();
                    path = ask_images_path();
                    cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                    draw_matches = ask_binary("YES", "NO");
                    cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                    save_results = ask_binary("YES", "NO");

                    single_image_descriptors(path, Flags::TRANSFORMED, Flags::SIFT, Flags::AUTOMATIC, octave_layers, distance_ratio, draw_matches, save_results);

                }
            //enters manual mode
                else{

                    clear_terminal();
                    cout<<"\033[0;36mYou have selected: Transformed Patches, Single Corrupted Image, Descriptor: SIFT, MAtcher: BFMatcher L2_NORM,  Manual RANSAC\033[0m"<<endl<<endl;
                    
                    octave_layers = ask_octave_layers();
                    distance_ratio = ask_distance_ratio();
                    path = ask_images_path();
                    cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                    draw_matches = ask_binary("YES", "NO");
                    cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                    save_results = ask_binary("YES", "NO"); 

                    single_image_descriptors(path, Flags::TRANSFORMED, Flags::SIFT, Flags::MANUAL, octave_layers, distance_ratio, draw_matches, save_results);
                 
                }
            }
        //entering ORB mode
            else{

                cout<<"\033[1;33mHow do you want to calculate Homography matrix?\033[0m"<<endl;
                homography_type = ask_binary("CV Library Function", "Manual RANSAC Function");

            //enters automatic mode
                if(homography_type){

                    clear_terminal();
                    cout<<"\033[0;36mYou have selected: Transformed Patches, Single Corrupted Image, Descriptor: ORB, MAtcher: BFMatcher HEMMING_NORM,  CV RANSAC\033[0m"<<endl<<endl;
                    
                    distance_ratio = ask_distance_ratio();
                    path = ask_images_path();
                    cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                    draw_matches = ask_binary("YES", "NO");
                    cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                    save_results = ask_binary("YES", "NO");

                    single_image_descriptors(path, Flags::TRANSFORMED, Flags::ORB, Flags::AUTOMATIC, octave_layers, distance_ratio, draw_matches, save_results);

                }
            //enters manual mode
                else{

                    clear_terminal();
                    cout<<"\033[0;36mYou have selected: Transformed Patches, Single Corrupted Image, Descriptor: ORB, MAtcher: BFMatcher HEMMING_NORM,  Manual RANSAC\033[0m"<<endl<<endl;
                    
                    distance_ratio = ask_distance_ratio();
                    path = ask_images_path();
                    cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                    draw_matches = ask_binary("YES", "NO");
                    cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                    save_results = ask_binary("YES", "NO"); 

                    single_image_descriptors(path, Flags::TRANSFORMED, Flags::ORB, Flags::MANUAL, octave_layers, distance_ratio, draw_matches, save_results);
                 
                }
            }
        }   
    }
//enters multiple Images mode
    else{

        cout<<"\033[1;33mWhat type of patches do you want to use?\033[0m"<<endl;
        patches_type = ask_binary("Basic Patches", "Transformed Patches [more computationally expensive]");

    //enters basic patches mode
        if(patches_type){

            cout<<"\033[1;33mHow do you want to calculate Homography matrix?\033[0m"<<endl;
            homography_type = ask_binary("CV Library Function", "Manual RANSAC Function");

        //enters automatic mode
            if(homography_type){

                clear_terminal();
                cout<<"\033[0;36mYou have selected: Basic Patches, Multiple Corrupted Images, Descriptor: SIFT, Matcher: BFMatcher L2_NORM,  CV RANSAC\033[0m"<<endl<<endl;

                octave_layers = ask_octave_layers();
                distance_ratio = ask_distance_ratio();
                tie(match_distance_threshold, sparsity_ratio_threshold) = ask_multiple_threshold();
                path = ask_images_path();
                cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                draw_matches = ask_binary("YES", "NO");
                cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                save_results = ask_binary("YES", "NO");

                multiple_images(path, Flags::BASE, Flags::AUTOMATIC, octave_layers, distance_ratio, match_distance_threshold, sparsity_ratio_threshold, draw_matches, save_results);   

            }
        //enters manual mode
            else{

                clear_terminal();
                cout<<"\033[0;36mYou have selected: Basic Patches, Multiple Corrupted Images, Descriptor: SIFT, MAtcher: BFMatcher L2_NORM,  Manual RANSAC\033[0m"<<endl<<endl;
            
                octave_layers = ask_octave_layers();
                distance_ratio = ask_distance_ratio();
                tie(match_distance_threshold, sparsity_ratio_threshold) = ask_multiple_threshold();
                path = ask_images_path();
                cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                draw_matches = ask_binary("YES", "NO");
                cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                save_results = ask_binary("YES", "NO");

                multiple_images(path, Flags::BASE, Flags::MANUAL, octave_layers, distance_ratio, match_distance_threshold, sparsity_ratio_threshold, draw_matches, save_results);

            }
        }
    //enters transformed patches mode
        else{

            cout<<"\033[1;33mHow do you want to calculate Homography matrix?\033[0m"<<endl;
            homography_type = ask_binary("CV Library Function", "Manual RANSAC Function");

        //enters automatic mode
            if(homography_type){

                clear_terminal();
                cout<<"\033[0;36mYou have selected: Transformed Patches, Multiple Corrupted Images, Descriptor: SIFT, MAtcher: BFMatcher L2_NORM,  CV RANSAC\033[0m"<<endl<<endl;
                
                octave_layers = ask_octave_layers();
                distance_ratio = ask_distance_ratio();
                tie(match_distance_threshold, sparsity_ratio_threshold) = ask_multiple_threshold();
                path = ask_images_path();
                cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                draw_matches = ask_binary("YES", "NO");
                cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                save_results = ask_binary("YES", "NO");

                multiple_images(path, Flags::TRANSFORMED, Flags::AUTOMATIC, octave_layers, distance_ratio, match_distance_threshold, sparsity_ratio_threshold, draw_matches, save_results);


            }
        //enters manual mode
            else{

                clear_terminal();
                cout<<"\033[0;36mYou have selected: Transformed Patches, Multiple Corrupted Images, Descriptor: SIFT, MAtcher: BFMatcher L2_NORM,  Manual RANSAC\033[0m"<<endl<<endl;

                octave_layers = ask_octave_layers();
                distance_ratio = ask_distance_ratio();
                tie(match_distance_threshold, sparsity_ratio_threshold) = ask_multiple_threshold();
                path = ask_images_path();
                cout<<"\033[1;33mDo you want to see intermediate results?\033[0m"<<endl;
                draw_matches = ask_binary("YES", "NO");
                cout<<"\033[1;33mDo you want to save the final results?\033[0m"<<endl;
                save_results = ask_binary("YES", "NO");

                multiple_images(path, Flags::TRANSFORMED, Flags::MANUAL, octave_layers, distance_ratio, match_distance_threshold, sparsity_ratio_threshold, draw_matches, save_results);

            }
        }
    }

}

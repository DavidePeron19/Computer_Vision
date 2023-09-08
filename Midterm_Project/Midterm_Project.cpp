

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/*
    GLOBAL VARIABLE DEFINITION
*/
//definiton of the image matrix to be used 
Mat image_in;
//definition of the global variables used for the trackbars
int median_kernel_size = 5;
int gaussian_kernel_size = 5;
int gaussian_sigmas = 1;
int bilateral_kernel_size = 5;
int bilateral_sigma_range = 1;
int bilateral_sigma_space = 1;


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


//I modified the original function given to be able to display the different images' histograms 
//togather without beeing overwritten
//I added a String input which is the name of the image for which we are going to plot the histograms
void showHistogram(std::vector<cv::Mat>& hists, String name)
{
    // Min/Max computation
    double h_max[3] = { 0,0,0 };
    double min;
    cv::minMaxLoc(hists[0], &min, &h_max[0]);
    cv::minMaxLoc(hists[1], &min, &h_max[1]);
    cv::minMaxLoc(hists[2], &min, &h_max[2]);

    //this is where my modification takes place, I added the name input to the
    //name of the window that the function will use to plot the histograms in the end
    
    std::string wname[3] = { name+"_Blue", name+"_Green", name+"_Red" };
    cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                             cv::Scalar(0,0,255) };

    std::vector<cv::Mat> canvas(hists.size());

    // Display each histogram in a canvas
    for (int i = 0, end = hists.size(); i < end; i++)
    {
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
        {
            cv::line(
                canvas[i],
                cv::Point(j, rows),
                cv::Point(j, rows - (hists[i].at<float>(j) * rows / h_max[i])),
                hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
                1, 8, 0
            );
        }
        cv::imshow(hists.size() == 1 ? "Value" : wname[i], canvas[i]);
    }
}


//function that will be passed to the trackbars and that is responible for updating
//the median filter and also showing the filtered image
void median_blur(int, void*) {

    // definition of the filtered image matrix
    Mat medianBlurrImag;

    //check if the kernel size is odd if not then make it odd by adding 1
    //I added 1 also to be sure we never use kernel size equal to 0
    if (median_kernel_size % 2 == 0) {
        median_kernel_size++;
    }

    //write on the console the exact values of the parameters used since the trackbar
    //does not represent accurate values (I was not able to set the trackbar to display only odd values)
    clear_terminal();
    cout << "median kernel size: " + to_string(median_kernel_size) + "\n";

    //apply the median blur filter
    medianBlur(image_in, medianBlurrImag, median_kernel_size);

    // Show the filtered image
    imshow("Median Blur", medianBlurrImag);
}

//function that will be passed to the trackbars and that is responible for updating
//the gaussian filter and also showing the filtered image
void gauss_blur(int, void*) {

    // definition of the filtered image matrix
    Mat gaussBlurred;

    //check if the kernel size is odd if not then make it odd by adding 1
    //I added 1 also to be sure we never use kernel size equal to 0
    if (gaussian_kernel_size % 2 == 0) {
        gaussian_kernel_size++;
    }

    //defining double values to to sigma since the trackbar can pass only integers 
    double sigma_double = gaussian_sigmas;
    //dividing the value given by the trackbar by 10 to obtain fractional values fot tthe parameter
    sigma_double /= 10;
    clear_terminal();

    //write on the console the exact values of the parameters used since the trackbar
    //does not represent accurate values (I was not able to set the trackbar to display 
    //only odd values and also it cannot display fractional values)
    cout << "gussian kernel size: " + to_string(gaussian_kernel_size)+"\n";
    cout << "gussian sigma: " + to_string(sigma_double) + "\n";

    //apply gaussian blur filter
    GaussianBlur(image_in, gaussBlurred, Size(gaussian_kernel_size,gaussian_kernel_size), sigma_double, sigma_double);

    // Show the filtered image
    imshow("Gaussian Blur", gaussBlurred);
}

//function that will be passed to the trackbars and that is responible for updating
//the bilateral filter and also showing the filtered image
 void bilateral_filt(int, void* ) {

     // definition of the filtered image matrix
     Mat bilateralBlurred;

     // check if the kernel size is odd if not then make it odd by adding 1
     //I added 1 also to be sure we never use kernel size equal to 0
     if (bilateral_kernel_size % 2 == 0) {
         bilateral_kernel_size++;         
     }
     //defining double values to to sigma since the trackbar can pass only integers 
     double sigma_double_r = bilateral_sigma_range;
     //dividing the value given by the trackbar by 10 to obtain fractional values fot tthe parameter
     sigma_double_r /= 10;
     //defining double values to to sigma since the trackbar can pass only integers 
     double sigma_double_s = bilateral_sigma_space;
     //dividing the value given by the trackbar by 10 to obtain fractional values fot tthe parameter
     sigma_double_s /= 10;

     //write on the console the exact values of the parameters used since the trackbar
     //does not represent accurate values (I was not able to set the trackbar to display 
     //only odd values and also it cannot display fractional values)
     clear_terminal();
     cout << "bilateral kernel size: " + to_string(bilateral_kernel_size) + "\n";
     cout << "bilateral sigma range: " + to_string(sigma_double_r) + "\n";
     cout << "bilateral sigma range: " + to_string(sigma_double_s) + "\n";

     // apply Bilateral filter to the image
     bilateralFilter(image_in, bilateralBlurred, bilateral_kernel_size, bilateral_sigma_range, bilateral_sigma_space);

     // Show the filtered image
     imshow("Bilateral Filter", bilateralBlurred);
 }


 //this function is used to calculate and show the histograms
 //it takes in input a vector of Mat elements that represents the plnes of the image
 //and the name of the image we want to display
 //it uses showHistogram to display the histograms
 //i decide to make this function to make the code more readable in the main
 void calc_show_hist(vector<Mat> planes, String name) {
     
     //parameters necessary to implement the calcHist functions
     int histSize = 256;
     float range[] = { 0, 256 }; //the upper boundary is exclusive
     const float* histRange[] = { range };
     bool uniform = true, accumulate = false;
     
     //definition of the vector containing the histograms
     vector <Mat> hists(3);

     //calculation of the histograms for the 3 planes
     calcHist(&planes[0], 1, 0, Mat(), hists[0], 1, &histSize, histRange, uniform, accumulate);
     calcHist(&planes[1], 1, 0, Mat(), hists[1], 1, &histSize, histRange, uniform, accumulate);
     calcHist(&planes[2], 1, 0, Mat(), hists[2], 1, &histSize, histRange, uniform, accumulate);
     
     //use showHistogram to diplay the histograms on thei window
     showHistogram(hists, name);
 }

int main()
{
    /*
                     PART 1: HISTOGRAM EQUALIZATION   
    */


    //getting as input the string containing the image path
    String path;
    cout << "insert path of the image you want to display: ";
    cin >> path;

    //reading the image from the path in input
    image_in = imread(path);

    //show the original image
    imshow("original image", image_in);

    // Wait for key press to advance
    waitKey(0);

    //defining the bgr planes of the image and then splitting it in the 3 components
    vector<Mat> bgr_planes;
    split(image_in, bgr_planes);
   
    //calculating and printing the histograms of the original image
    calc_show_hist(bgr_planes, "bgr image");

    // Wait for key press to advance
    waitKey(0);
    
    //                  BGR EQUALIZATION

    //definition of the vector that will contain the bgr equalized planes
    vector<Mat> bgr_planes_equalized(3);
  
    //equalizing every component of of the image
    equalizeHist(bgr_planes[0],bgr_planes_equalized[0]);
    equalizeHist(bgr_planes[1], bgr_planes_equalized[1]);
    equalizeHist(bgr_planes[2], bgr_planes_equalized[2]);
    
    //defining a new Mat that will contain the new equalized image
    Mat image_out;

    //merging the equalized planes into an image
    merge(bgr_planes_equalized, image_out);

    //show the equalized image
    imshow("imgage Equalized", image_out);
    // Wait for key press to advance
    waitKey(0);
    
    //calculating and printing the histograms of the equalized image
    calc_show_hist(bgr_planes_equalized, "bgr equalized image");

    // Wait for key press to advance
    waitKey(0);
    
    //                    Lab EQUALIZATION

    //definining a new Mat that will contain the Lab image
    Mat image_lab;

    //converting the bgr image to Lab
    cvtColor(image_in, image_lab, COLOR_BGR2Lab);

    //defining the Lab planes of the image and then splitting it in the 3 components
    vector<Mat> lab_planes;
    split(image_lab, lab_planes);

    //equalizing only the firt component (luminance) of the Lab components
    equalizeHist(lab_planes[0], lab_planes[0]);

    //defining a new Mat that will contain the new equalized Lab image
    Mat image_out_lab;
    
    //merging the Lab planes into an image
    merge(lab_planes, image_out_lab);

    //converting back the Lab image to bgr
    cvtColor(image_out_lab, image_out_lab, COLOR_Lab2BGR);

    //show the image with the luminance equalized
    imshow("imgage Equalized Lab", image_out_lab);

    // Wait for key press to advance
    waitKey(0);

    //defining the vector that will conatin the bgr components of the image with
    //the luminance equalized and then splitting into the 3 components
    vector<Mat> lab_equalized_planes;
    split(image_out_lab, lab_equalized_planes);

    //calculating and printing the histograms of the liminance equalized image
    calc_show_hist(lab_equalized_planes, "Lab equalized image");

    // Wait for key press to advance
    waitKey(0);

    //close all windows opened in this part of the homework
    destroyAllWindows();

       
    /*
                         PART 2: IMAGE FILTERING
    */


    //defining the max value that can be displayed in the trackbars for kernel sizes and sigmas
    int max_kernel_size = 100; 
    //sigma will be devided by 10 so the real inteval will be [0, 100] with step 0.1
    int max_sigma = 1000;


    //                       MEDIAN FILTER
  
    //define name of the window that will display medain filtering
    namedWindow("Median Blur", WINDOW_AUTOSIZE);

    //creating the trackbar to control kernel size of median filter and call the function
    //that will realize median filtering
    createTrackbar("Kernel Size", "Median Blur", &median_kernel_size, max_kernel_size, median_blur);
    // Wait for key press to advance
    waitKey(0);

    //                      GUASSIAN FILTER

    //define name of the window that will display gaussian filtering
    namedWindow("Gaussian Blur", WINDOW_AUTOSIZE);

    //creating the trackbars to control kernel size and sigmas of gaussian filter 
    //and call the function that will realize gaussian filtering
    createTrackbar("Kernel Size", "Gaussian Blur", &gaussian_kernel_size, max_kernel_size, gauss_blur);
    createTrackbar("Sigma", "Gaussian Blur", &gaussian_sigmas, max_sigma, gauss_blur);

    // Wait for key press to advance
    waitKey(0);

    //                      BILATERAL FILTER
    
    //define name of the window that will display gaussian filtering
    namedWindow("Bilateral Filter", WINDOW_AUTOSIZE);

    //creating the trackbars to control kernel size and sigmas of bilateral filter 
    //and call the function that will realize bilateral filtering
    createTrackbar("kernel_sz", "Bilateral Filter", &bilateral_kernel_size, max_kernel_size, bilateral_filt);
    createTrackbar("sigma_r", "Bilateral Filter", &bilateral_sigma_range, max_sigma, bilateral_filt);
    createTrackbar("sigma_s", "Bilateral Filter", &bilateral_sigma_space, max_sigma, bilateral_filt);
    
    // Wait for key press to advance
    waitKey(0);
}

/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <list>
#include <deque>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

//**Student create pop_front vector
template <typename T>
void pop_front(std::vector<T> &v)
{
    if (v.size() > 0)
    {
        v.erase(v.begin());
    }
}
/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{     // Create csv file with date and time
    time_t t_current;
    struct tm* t_info;
    char buf[100];
    string fn_output;

    time(&t_current); // Get time now
    t_info = localtime(&t_current);

    strftime(buf, 100, "%Y_%m_%d_%H_%M_%S.csv", t_info );
    fn_output = string(buf);

    ofstream outfile(fn_output);

    // Create file header
    outfile << "Detector, Descriptor, Matcher, Selector, Number of Keypoints, Size of Neighborhood, Number of Matched Keypoints, Time (ms Keypoints Detection), Time (ms Descriptor Extraction), Time (ms Total)"<< endl;


    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);
        // Remove the old frame (front frame ), if dataBuffer bigger dataBufferSize setup
        if (dataBuffer.size() > dataBufferSize)
        {
            pop_front(dataBuffer);
            cout << "Remove the front image" << endl;
        }
        //If dataBuffer less than dataBufferSize
        else
        {
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
        }

        //// EOF STUDENT ASSIGNMENT

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        // string detectorType = "SHITOMASI";
        // string detectorType = "HARRIS";
        // string detectorType = "FAST";
        // string detectorType = "BRISK";
         string detectorType = "ORB";
        // string detectorType = "AKAZE";
        // string detectorType = "SIFT";

        //  TASK MP.9 detector timer 
        double detector_timer = (double)cv::getTickCount();

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST") == 0 ||
                 detectorType.compare("BRISK") == 0 ||
                 detectorType.compare("ORB") == 0 ||
                 detectorType.compare("AKAZE") == 0 ||
                 detectorType.compare("SIFT") == 0)
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        else
        { // checking if not find any detector
            throw invalid_argument("Not found detector type: " + detectorType);
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            //Filter the key  point inside the preceding vehicle
            vector<cv::KeyPoint> keypoint_filter;
            for (auto kp : keypoints)
            {
                if (vehicleRect.contains(kp.pt))
                {
                    keypoint_filter.push_back(kp);
                }
            }
            keypoints = keypoint_filter;

        }
        detector_timer = ((double)cv::getTickCount() - detector_timer) / cv::getTickFrequency();
        cout << "| "<<detectorType << "  |   " << keypoints.size() << "  |   " << 1000 * detector_timer / 1.0 << "   |   " << endl;

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS dones " << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
         string descriptorName = "BRISK"; 
        // string descriptorName = "BRIEF";
        // string descriptorName = "ORB";
        // string descriptorName = "FREAK";
        // string descriptorName = "AKAZE";  // Fails with all non-AKAZE detectors
        // string descriptorName = "SIFT";  // Fails with ORB detectors

        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorName);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
             //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
             //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
           
            

            double descriptor_timer = (double)cv::getTickCount();

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);
            
            descriptor_timer= ((double)cv::getTickCount() - descriptor_timer) / cv::getTickFrequency();

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
            cout << "descriptorName:" + descriptorName << ",";
            cout << "descriptorType:" + descriptorType << ",";
            cout << "Num matches point" << ":";
            cout << matches.size() << ",";
            cout << "timer in: "<< 1000 * descriptor_timer/ 1.0 <<  " ms" << "," << endl;
            
            outfile << detectorType << ", " << descriptorName << ", " << matcherType << ", " << selectorType << ", " << keypoints.size()  << ", " << keypoints[0].size  << ", " << matches.size()  << ", " << 1000 * detector_timer / 1.0  << ", " << 1000 * descriptor_timer / 1.0  << ", " <<1000 * (detector_timer +descriptor_timer) / 1.0 << endl;

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    outfile.close();
    return 0;
}
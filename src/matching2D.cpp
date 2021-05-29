#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // Implement Brute force Matching
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType; 
        // With SIFT 
        if (descriptorType.compare("DES_HOG") == 0)
        {
            normType = cv::NORM_L2;
        }
        // with another binary descriptor
        else if (descriptorType.compare("DES_BINARY") == 0)
        {
            normType = cv::NORM_HAMMING;
        }
        else
        {
        // checking if not find any descriptorType
            throw invalid_argument("Not found descriptorType " + descriptorType);
        }
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }

    // Implement FLANN matching 
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // with SIFT
        if (descriptorType.compare("DES_HOG") == 0)
        {
            matcher = cv::FlannBasedMatcher::create();
        }

        // with all other binary descriptorTypes
        else if (descriptorType.compare("DES_BINARY") == 0)
        {
            const cv::Ptr<cv::flann::IndexParams>& indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
            matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
        }
         else
        {
        // checking if not find any descriptorCatergory
            throw invalid_argument("Not found descriptorType " + descriptorType);
        }
    }
    else {
        throw invalid_argument("Not a valid matcherType" + matcherType);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }

    //Implement k nearest neighbors (k=2)
    else if (selectorType.compare("SEL_KNN") == 0)
    { 

        int k = 2;
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        
        // Filter matches using descriptor distance ratio test 
        double minDescDistRatio = 0.8;
        for (auto it : knn_matches) {
            // The returned knn_matches vector contains some nested vectors with size < 2 !?
            if ( 2 == it.size() && (it[0].distance < minDescDistRatio * it[1].distance) ) {
                matches.push_back(it[0]);
            }
        }
    }
    else {
        throw invalid_argument("Not a valid selectorType" + selectorType);
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::SIFT::create();
    }
    else
    {
        // checking if not find any descriptor
            throw invalid_argument("Not found descriptor type: " + descriptorType);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << keypoints.size() << " n keypoints " << "in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // Cout for markdown
    //cout << "| Shi-Tomasi" << "  |   " << keypoints.size() << "  |   " << 1000 * t / 1.0 << "   |   " << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
//Task 2. Adding detKeypointsHarris 
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    //set initial time
    double t = (double)cv::getTickCount();
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    if (bVis) {
        string windowName = "Harris Corner Detector Response Matrix";
        cv::namedWindow(windowName, 4);
        cv::imshow(windowName, dst_norm_scaled);
        cv::waitKey(0);
    }


    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
     t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
     // Output the number of key point
    //cout << "detection Harris " << " with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    // cout for markdown
    //cout << "|  Harris " << "  |   " << keypoints.size() << "  |   " << 1000 * t / 1.0 << "   |   " << endl;
    // visualize keypoints
    if (bVis) {
        string windowName = "Harris Corner Detection Show";
        cv::namedWindow(windowName, 5);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
    // EOF STUDENT CODE
}

//// Task2. Adding detkeypointsMordern
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
    double t = (double)cv::getTickCount();
    if (detectorType.compare("FAST") == 0)
    {
        auto fast = cv::FastFeatureDetector::create(); 
        fast -> detect(img, keypoints);
    }
    else if (detectorType.compare("BRISK") == 0) {
        auto brisk = cv::BRISK::create(); 
        brisk -> detect(img, keypoints);
    }
    else if (detectorType.compare("ORB") == 0) {
        auto orb = cv::ORB::create(); 
        orb -> detect(img, keypoints);
    }
    else if (detectorType.compare("AKAZE") == 0) {
        auto akaze = cv::AKAZE::create(); 
        akaze -> detect(img, keypoints);
    }
    else if (detectorType.compare("SIFT") == 0) {
        auto sift = cv::SIFT::create(); 
        sift -> detect(img, keypoints);
    }
    else
    {

        //  invalid argument error
        throw invalid_argument("Not found detector type: " + detectorType);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
     // Output the number of key point
    //cout << "detection: " << detectorType << " with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // cout for markdown
    // cout << "| "<<detectorType << "  |   " << keypoints.size() << "  |   " << 1000 * t / 1.0 << "   |   " << endl;
    // visualize keypoints
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Keypoints Detection Results";
        cv::namedWindow(windowName, 2);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

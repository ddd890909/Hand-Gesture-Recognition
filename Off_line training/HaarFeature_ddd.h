//
//HaarFeature_ddd.h
//Hand gesture recognition
//Haar-like feature extraction from IHG dataset for matlab to train HMM model
//
//Destin Liu
//2015.3.22

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string> 
#include <cstdio>
#include <Windows.h>
#include <math.h>
#include <io.h>
#include "CT_ddd.h"

using namespace std;
using namespace cv;

void draw_frameCount(Mat Frame,int framecount);
void skinExtract(const Mat &frame, Mat &skinArea);
void hand_detect(Mat& frameImage, Rect& rect);
void hand_track(string path_video, string name_window, string path_video_CT, string path_featureValue, CompressiveTracker &ct);

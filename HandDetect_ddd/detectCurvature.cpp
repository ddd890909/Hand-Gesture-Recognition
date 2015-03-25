
#include "opencv2/opencv.hpp"    
    
using namespace cv;    
using namespace std;  

//#define detectCurvature
#ifdef detectCurvature

void skinExtract(const Mat &frame, Mat &skinArea);    
    
string save_path = "datas";

int main(int argc, char* argv[])    
{    
    Mat frame, skinArea, show_img,skinAreaOpen;    
    //VideoCapture capture;    
    //
    //capture.open(0);    
    //if (!capture.isOpened())    
    //{    
    //    cout<<"No camera!\n"<<endl;    
    //    return -1;    
    //}    
    
    while (1)    
    {    
        //capture >> frame;    
        Mat frame = imread("D:\\lab\\KinectRecord\\ddd\\Record_ddd\\Record_ddd\\datas\\images\\ddd_cImageBGR_1.jpg"); 

        if (frame.empty())    
            break;    
    
        GaussianBlur(frame, frame, Size(3, 3), 0);    
        skinArea.create(frame.rows, frame.cols, CV_8UC1);    
        skinExtract(frame, skinArea);       
        frame.copyTo(show_img, skinArea);    
    
        vector<vector<Point> > contours;    
        vector<Vec4i> hierarchy;    
    
		morphologyEx(skinArea,skinAreaOpen,MORPH_OPEN,Mat(3,3,CV_8U),Point(-1,-1),1);

        //寻找轮廓    
        findContours(skinAreaOpen, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);    
    
        // 找到最大的轮廓    
        int index;    
        double area, maxArea(0);    
        for (int i=0; i < contours.size(); i++)    
        {    
            area = contourArea(Mat(contours[i]));    
            if (area > maxArea)    
            {    
                maxArea = area;    
                index = i;    
            }               
        }    
    
        drawContours(frame, contours, index, Scalar(0, 0, 255), 2, 8, hierarchy );    
            
		//find center
        //Moments moment = moments(skinArea, true);    
        //Point center(moment.m10/moment.m00, moment.m01/moment.m00); 

		Rect rect=boundingRect(contours[index]);
		rectangle(frame,rect,Scalar(255,255,255),2);

		imshow("frame", frame); 

		Point center( rect.x+rect.width/2, rect.y+rect.height/3 );
        circle(show_img, center, 8 ,Scalar(0, 0, 255), CV_FILLED);  
    
        // 寻找指尖    
        vector<Point> couPoint = contours[index];    
        int max(0), count(0), notice(0);    
        vector<Point> fingerTips;    
        Point p, q, r;    
        for (int i = 5; (i < (couPoint.size() - 5)) && couPoint.size(); i++)    
        {    
            q = couPoint[i - 5];    
            p = couPoint[i];    
            r = couPoint[i + 5];    
            int dot = (q.x - p.x ) * (q.y - p.y) + (r.x - p.x ) * (r.y - p.y);    
            if (dot < 5 && dot > -5)    
            {    
                int cross = (q.x - p.x ) * (r.y - p.y) - (r.x - p.x ) * (q.y - p.y);    
                if (cross > 0)    
                {    
                    fingerTips.push_back(p);    
                    circle(show_img, p, 5 ,Scalar(255, 0, 0), CV_FILLED);    
                    line(show_img, center, p, Scalar(255, 0, 0), 2);        
                }    
            }    
        }    
    
        imshow("show_img", show_img);    
    
        if ( cvWaitKey(20) == 'q' )    
            break;    
    }    
    
	//image write
    imwrite(save_path+"/"+"images"+"/"+"ddd_handDetect_Curvature.jpg", frame);
    imwrite(save_path+"/"+"images"+"/"+"ddd_fingerDetect_Curvature.jpg", show_img);

    return 0;    
}    
    
//肤色提取，skinArea为二值化肤色图像    
void skinExtract(const Mat &frame, Mat &skinArea)    
{    
    Mat YCbCr;    
    vector<Mat> planes;    
    
    //转换为YCrCb颜色空间    
    cvtColor(frame, YCbCr, CV_RGB2YCrCb);    
    //将多通道图像分离为多个单通道图像    
    split(YCbCr, planes);     
    
    //运用迭代器访问矩阵元素    
    MatIterator_<uchar> it_Cb = planes[1].begin<uchar>(),    
                        it_Cb_end = planes[1].end<uchar>();    
    MatIterator_<uchar> it_Cr = planes[2].begin<uchar>();    
    MatIterator_<uchar> it_skin = skinArea.begin<uchar>();    
    
    //人的皮肤颜色在YCbCr色度空间的分布范围:100<=Cb<=127, 138<=Cr<=170    
    for( ; it_Cb != it_Cb_end; ++it_Cr, ++it_Cb, ++it_skin)    
    {    
        if (138 <= *it_Cr &&  *it_Cr <= 170 && 100 <= *it_Cb &&  *it_Cb <= 127)    
            *it_skin = 255;    
        else    
            *it_skin = 0;    
    }    
    
    //膨胀和腐蚀，膨胀可以填补凹洞（将裂缝桥接），腐蚀可以消除细的凸起（“斑点”噪声）    
    dilate(skinArea, skinArea, Mat(5, 5, CV_8UC1), Point(-1, -1));    
    erode(skinArea, skinArea, Mat(5, 5, CV_8UC1), Point(-1, -1));    
}

#endif

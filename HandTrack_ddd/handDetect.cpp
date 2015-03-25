
#include "opencv2/opencv.hpp"    
    
using namespace cv;    
using namespace std;    


void skinExtract(const Mat &frame, Mat &skinArea);

void handDetect(Mat &frameImage, Rect &rect)
{    
    Mat show_img, skinArea, skinAreaErode, skinAreaDilate,skinAreaOpen, skinAreaClose; 


    skinArea.create(frameImage.rows, frameImage.cols, CV_8UC1);    
    skinExtract(frameImage, skinArea);
	  
    //frameImage.copyTo(show_img, skinArea);    
	frameImage.copyTo(show_img); 

    vector< vector<Point> > contours;    
    vector<Vec4i> hierarchy;    
    
	//形態學
	//erode(skinArea,skinAreaErode,Mat(5,5,CV_8U),Point(-1,-1),1);
	//dilate(skinArea, skinAreaDilate,Mat(5,5,CV_8U),Point(-1,-1),1);
	//morphologyEx(skinArea,skinAreaOpen,MORPH_OPEN,Mat(3,3,CV_8U),Point(-1,-1),1);
	//morphologyEx(skinArea,skinAreaClose,MORPH_CLOSE,Mat(3,3,CV_8U),Point(-1,-1),1);

    //寻找轮廓    
    findContours(skinArea, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);    

    //找到最大的轮廓    
    int index1, index2, index3;    
    double area, maxArea1(0);
	//double maxArea2(0), maxArea3(0);

	for (int i=0; i < contours.size(); i++)    
    {    
        area = contourArea(Mat(contours[i]));    
        if (area > maxArea1)    
        {    
            maxArea1 = area;    
            index1 = i;
        } 
		//if ( (area > maxArea2) && (area < maxArea1) )
        //{    
        //    maxArea2 = area;    
        //    index2 = i;
        //}
		//if ( (area > maxArea3) && (area < maxArea2) )
        //{    
        //    maxArea3 = area;    
        //    index3 = i;
        //} 
    }    
    
    drawContours(frameImage, contours, index1, Scalar(0, 0, 255), 2, 8, hierarchy );    
        
	//find center
    //Moments moment = moments(skinArea, true);    
    //Point center(moment.m10/moment.m00, moment.m01/moment.m00); 

	//int xMax=frame.cols;
	//int xMin=0;
	//int yMax=frame.rows;
	//int yMin=0;

	//draw rectangle
	Rect rectOri=boundingRect(contours[index1]);

	rect.x=rectOri.x;
	rect.y=rectOri.y;
	rect.width=rectOri.width+10;
	rect.height=rectOri.height+10;
	//rectangle(frameImage,rect,Scalar(255,255,255),2);
	printf("rect.x=%d\n",rect.x);
	printf("rect.y=%d\n",rect.y);
	printf("rect.width=%d\n",rect.width);
	printf("rect.height=%d\n",rect.height);

	//imshow("frameImage", frameImage); 

	//Point center( rect.x+rect.width/2, rect.y+rect.height/3 );
    //circle(show_img, center, 8 ,Scalar(0, 0, 255), CV_FILLED);
    
    // 寻找指尖   
	/*
    vector<Point> couPoint = contours[index1];    
    vector<Point> fingerTips;    
    Point tmp;    
    int max(0), count(0), notice(0);    
    for (int i = 0; i < couPoint.size(); i++)    
    {
        tmp = couPoint[i];    
        int dist = (tmp.x - center.x) * (tmp.x - center.x) + (tmp.y - center.y) * (tmp.y - center.y);    
        if (dist > max)    
        {    
            max = dist;    
            notice = i;    
        }    

        // 计算最大值保持的点数，如果大于40（这个值需要设置，本来想根据max值来设置,但是不成功，不知道为何），那么就认为这个是指尖    
        if (dist != max)    
        {    
            count++;    
            if (count > 30)    
            {    
                count = 0;    
                max = 0;    
                bool flag = false;    
                // 高于手心的点不算    
                if (center.y > couPoint[notice].y )    
                    continue;    
                // 离得太近的不算    
                for (int j = 0; j < fingerTips.size(); j++)    
                {    
                    if (abs(couPoint[notice].x - fingerTips[j].x) < 20)    
                    {    
                        flag = true;    
                        break;    
                    }    
                }    
                if (flag) continue;
                fingerTips.push_back(couPoint[notice]);    
                circle(show_img, couPoint[notice], 6 ,Scalar(0, 255, 0), CV_FILLED);    
                line(show_img, center, couPoint[notice], Scalar(255, 0, 0), 2);                 
            }    
        }    
    }    
	*/

    return;    
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
    
	for (int i = 1; i < skinArea.cols; i++)
	{		
		for (int j = 1; j < skinArea.rows; j++)
		{
			if (i<180 || i>400 || j<230 || j>400)
			{
				skinArea.at<uchar>(j,i)=0;
			}
			//printf("j=%d\n",j);
		}		
		//printf("i=%d\n",i);
	}

    //膨胀和腐蚀，膨胀可以填补凹洞（将裂缝桥接），腐蚀可以消除细的凸起（“斑点”噪声）    
    dilate(skinArea, skinArea, Mat(5, 5, CV_8UC1), Point(-1, -1));    
    erode(skinArea, skinArea, Mat(5, 5, CV_8UC1), Point(-1, -1));    
}

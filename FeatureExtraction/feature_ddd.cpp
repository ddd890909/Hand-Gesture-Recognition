
//feature extraction from image sequences

#include <afxwin.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string> 
#include <cstdio>
#include <Windows.h>
#include "CompressiveTracker.h"
#include <math.h>
#include <io.h>

using namespace cv;
using namespace std;

//#define RGB_ddd
#define DEP_ddd

//Rect box_RGB,box_DEP; // tracking box
//bool drawing_box = false;
//bool gotBB = false;	// got tracking box or not
//string video;
bool fromfile = true; //bool fromfile = false;
int featureNum=50;
int	featureMinNumRect = 2;
int	featureMaxNumRect = 4;	// number of rectangle in a Haar from 2 to 4
int imageWidth=120;
int imageHeight=120;
vector<vector<Rect>> features;
vector<vector<float>> featuresWeight;
RNG rng;

//////////////////
void HaarFeature_ddd(int _numFeature)
/*Description: compute Haar features
  Arguments:
  -_objectBox: [x y width height] object rectangle
  -_numFeature: total number of features.The default is 50
*/
{
	//_numFeature是一个样本box的harr特征个数，共50个。而上面说到，  
    //每一个harr特征是由2到3个随机选择的矩形框（vector<Rect>()类型）来构成的。
	features = vector<vector<Rect>>(_numFeature, vector<Rect>());
	//每一个反应特征的矩形框对应于一个权重，实际上就是随机测量矩阵中相应的元素，用它来与对应的特征
    //相乘，表示以权重的程度来感知这个特征。换句话说，featuresWeight就是随机测量矩阵。  
    //这个矩阵的元素的赋值看第二部分。  
	featuresWeight = vector<vector<float>>(_numFeature, vector<float>());
	
	//numRect是每个特征的矩形框个数 or 随机测量矩阵中的s？or both？
    //s取2或者3时，矩阵就满足Johnson-Lindenstrauss推论。
	int numRect;
	Rect rectTemp;
	float weightTemp;
    
	for (int i=0; i<_numFeature; i++)
	{
		//如何生成服从某个概率分布的随机数（或者说 sample）的问题。  
        //比如，想要从一个服从正态分布的随机变量得到 100 个样本，那么肯定抽到接近其均值的样本的  
        //概率要大许多，从而导致抽到的样本很多是集中在那附近的。  
        //rng.uniform()返回一个从[ 1，2）范围均匀采样的随机数，即在[ 1，2）内服从均匀分布（取不同值概率相同）  
        //那么下面的功能就是得到[ 2，4）范围的随机数，然后用cvFloor返回不大于参数的最大整数值，那要么是2，要么是3。  
		numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));
	    
		//int c = 1;
		for (int j=0; j<numRect; j++)
		{
			//我在一个box中随机生成一个矩形框，那和你这个box的x和y坐标就无关了，但我必须保证我选择  
            //的这个矩形框不会超出你这个box的范围 
            //但这里的3和下面的2是啥意思呢？个人理解是为了避免这个矩形框太靠近box的边缘了  
            //要离边缘最小2个像素，不知道这样理解对不对
			rectTemp.x = cvFloor(rng.uniform(0.0, (double)(imageWidth - 3)));
			rectTemp.y = cvFloor(rng.uniform(0.0, (double)(imageHeight - 3)));
			//cvCeil 返回不小于参数的最小整数值
			rectTemp.width = cvCeil(rng.uniform(0.0, (double)(imageWidth - rectTemp.x - 2)));
			rectTemp.height = cvCeil(rng.uniform(0.0, (double)(imageHeight - rectTemp.y - 2)));
			//保存得到的特征模板
			features[i].push_back(rectTemp);

			//weightTemp = (float)pow(-1.0, c);
			//pow(-1.0, c)也就是-1的c次方，而c随机地取0或者1，也就是说weightTemp是随机的正或者负。  
            //随机测量矩阵中，矩阵元素有三种，sqrt(s)、-sqrt(s)和零。为正和为负的概率是相等的，  
            //这就是为什么是[2，4）均匀采样的原因，就是取0或者1概率一样。  
            //但是这里为什么是sqrt(s)分之一呢？还有什么时候是0呢？论文中是0的概率不是挺大的吗？  
            //没有0元素，哪来的稀疏表达和压缩呢？（当然稀疏表达的另一个好处就是只需保存非零元素） 
			weightTemp = (float)pow(-1.0, cvFloor(rng.uniform(0.0, 2.0))) / sqrt(float(numRect));
			
			//保存每一个特征模板对应的权重 
			featuresWeight[i].push_back(weightTemp);
		}
	}
}

void getFeatureValue_ddd(Mat& _imageIntegral, Mat& _sampleFeatureValue_ddd)
{
	_sampleFeatureValue_ddd.create(featureNum, 1, CV_32F);
	float tempValue;
	int xMin;
	int xMax;
	int yMin;
	int yMax;

	for (int i=0; i<featureNum; i++)
	{
		tempValue = 0.0f;
		for (size_t k=0; k<features[i].size(); k++)
		{
			//features中保存的特征模板（矩形框）是相对于box的相对位置的，  
            //所以需要加上box的坐标才是其在整幅图像中的坐标 
			xMin = features[i][k].x;
			xMax = features[i][k].x + features[i][k].width;
			yMin = features[i][k].y;
			yMax = features[i][k].y + features[i][k].height;
			//通过积分图来快速计算一个矩形框的像素和 
            //tempValue就是经过稀释矩阵加权后的灰度和
            //每一个harr特征是由2到3个矩形框来构成的，对这些矩形框的灰度加权求和  
            //作为这一个harr特征的特征值。然后一个样本有50个harr特征 
			tempValue += featuresWeight[i][k] * 
				(_imageIntegral.at<float>(yMin, xMin) +
				_imageIntegral.at<float>(yMax, xMax) -
				_imageIntegral.at<float>(yMin, xMax) -
				_imageIntegral.at<float>(yMax, xMin));
		}
		if(tempValue==0)
		{
			tempValue += 0.1;
		}
		_sampleFeatureValue_ddd.at<float>(0,i) = tempValue;
	}
}

void processFrame_ddd(Mat& _frame, Mat& detectFeatureValue_ddd)
{
	//计算这一帧的积分图
	Mat imageIntegral;  //图像的积分图
	integral(_frame, imageIntegral, CV_32F);
	//用积分图来计算上面采集到的每个box的haar特征  
	getFeatureValue_ddd(imageIntegral, detectFeatureValue_ddd);  //ddd计算目标box的haar特征
}

///////////////////
void draw_frameCount(Mat Frame,int framecount)//Frameframecount
{
    char strFrame[10];
    sprintf_s(strFrame, "#%0d ",framecount);
    putText(Frame,strFrame,cvPoint(10,10),2,0.5,CV_RGB(200,25,200));
}

int main(int argc, char * argv[])
{
	int frameNumber;

#ifdef RGB_ddd
#endif

#ifdef DEP_ddd

	Mat frame_DEP;
	Mat frame_DEP_gray;
	Mat featureValue_DEP_ddd;

	//init
	HaarFeature_ddd(featureNum);

	for(int currentSubject=1;currentSubject<=10;currentSubject++) //each subject, no more than 10
	{
		string folderName_img="D:\\lab\\KinectRecord\\ddd\\Record_ddd\\FeatureExtraction\\train_sequence\\gestureData\\ddd\\Sequence";
		string folderName_dat="D:\\lab\\KinectRecord\\ddd\\Record_ddd\\FeatureExtraction\\FeatureData\\gestureData\\ddd\\HaarFeatureValue\\matlab";
		
		for(int currentSequence=1;currentSequence<=36;currentSequence++) //each sequence, no more than 36
		{		
			char fileName_seq[1000];
			char fileName_dat[1000];
			if(currentSubject<10)
			{
				if(currentSequence<10)
				{
					sprintf_s(fileName_seq,"%s\\sub_depth_0%d_0%d",folderName_img.c_str(),currentSubject,currentSequence);
					sprintf_s(fileName_dat,"%s\\sub_depth_0%d_0%d.dat",folderName_dat.c_str(),currentSubject,currentSequence);
				}
				else
				{
					sprintf_s(fileName_seq,"%s\\sub_depth_0%d_%d",folderName_img.c_str(),currentSubject,currentSequence);
					sprintf_s(fileName_dat,"%s\\sub_depth_0%d_%d.dat",folderName_dat.c_str(),currentSubject,currentSequence);
				}
			}
			else //currentSubject=10
			{
				if(currentSequence<10)
				{
					sprintf_s(fileName_seq,"%s\\sub_depth_%d_0%d",folderName_img.c_str(),currentSubject,currentSequence);
					sprintf_s(fileName_dat,"%s\\sub_depth_%d_0%d.dat",folderName_dat.c_str(),currentSubject,currentSequence);
				}
				else
				{
					sprintf_s(fileName_seq,"%s\\sub_depth_%d_%d",folderName_img.c_str(),currentSubject,currentSequence);
					sprintf_s(fileName_dat,"%s\\sub_depth_%d_%d.dat",folderName_dat.c_str(),currentSubject,currentSequence);
				}
			}
	
			if( _access(fileName_seq,0)== -1 )
			{
				continue;
			}

			//FeatureValueWriter
			string HaarFeatureValue_DEP=fileName_dat;
			ofstream fout_DEP(HaarFeatureValue_DEP);

			for(int currentFrame=1;;currentFrame++) //each frame
			{
				//cout<< currentFrame <<endl;
				
				char fileName_img[1000];
				if(currentSubject<10)
				{
					if(currentSequence<10)
						sprintf_s(fileName_img,"%s\\sub_depth_0%d_0%d\\%d.jpg",folderName_img.c_str(),currentSubject,currentSequence,currentFrame);
					else
						sprintf_s(fileName_img,"%s\\sub_depth_0%d_%d\\%d.jpg",folderName_img.c_str(),currentSubject,currentSequence,currentFrame);
				}
				else //currentSubject=10
				{
					if(currentSequence<10)
						sprintf_s(fileName_img,"%s\\sub_depth_%d_0%d\\%d.jpg",folderName_img.c_str(),currentSubject,currentSequence,currentFrame);
					else
						sprintf_s(fileName_img,"%s\\sub_depth_%d_%d\\%d.jpg",folderName_img.c_str(),currentSubject,currentSequence,currentFrame);
				}
				
				//load path
				frame_DEP=imread(fileName_img,0);
				if(frame_DEP.empty())
				{
				  cout<< "only "<< currentFrame-1<< " frames"<< endl;
				  frameNumber=currentFrame;
				  break;
				}
				//draw framecount
			    //draw_frameCount(frame_DEP,currentFrame);

				//Draw haar_boxes
			    //for (int i=0; i<10; i++) //draw 10 boxes
				//{  
				//	rectangle(frame_DEP, features[i][1], Scalar(255,255,255),1);
				//}
				
				//display
				//imshow("frame_DEP",frame_DEP);
				//waitKey(0);

				// change to gray
				//cvtColor(frame_DEP, frame_DEP_gray, CV_RGB2GRAY);
				frame_DEP_gray=frame_DEP;
				// Process Frame
				processFrame_ddd(frame_DEP_gray, featureValue_DEP_ddd);
				
				//writer featureValue
				fout_DEP<< format(featureValue_DEP_ddd.t(),"csv")<<endl;
				//for(int a=0; a<featureValue_DEP_ddd.rows; a++){
				//	for(int b=0; b<featureValue_DEP_ddd.cols; b++){
				//		fout_DEP<<featureValue_DEP_ddd.at<float>(a,b)<<" ";
				//	}
				//}
				//fout_DEP<< endl;
			}
		}
	}

#endif

	return 0;
}
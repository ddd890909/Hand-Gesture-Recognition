
/***********Depth Sense 325*****
*******************************************************
Decription	: This is the record program for thesis <<hand>>
Author		: Chengyin Liu
*******************************************************/

#ifdef _MSC_VER
#include <windows.h>
#endif

// STL Header
#include <iostream>
#include <stdio.h>
#include <vector>
#include <exception>
  
// OpenCV Header
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
  
// DepthSense DS325 Header
#include <DepthSense.hxx>

// namespace
using namespace DepthSense;
using namespace std;
using namespace cv;

#define RECORD
string classnumber = "24";
string subjectnumber ="3";
//#define WRITE_IMG

Context g_context;
DepthNode g_dnode;
ColorNode g_cnode;
AudioNode g_anode;

uint32_t g_aFrames = 0;
uint32_t g_cFrames = 0;
uint32_t g_dFrames = 0;

bool g_bDeviceFound = false;
ProjectionHelper* g_pProjHelper = NULL;
StereoCameraParameters g_scp;

//default width and height
int video_size_width = 640;
int video_size_height = 480;
int video_size_fps = 30;

//save path
string save_path = "D://lab//KinectRecord//ddd//Record_ddd//data_train//HandGesture_ddd//Class_"+classnumber;
//string data_name = "demo";

//char g_aFramesChar='0'+g_aFrames;
char g_cFramesChar='0'+g_cFrames;
char g_dFramesChar='0'+g_dFrames;

Mat imageColorResizePub;

//Record the data, if you want to
#ifdef RECORD	 	
    //initialize the VideoWriter object
	VideoWriter VideoWriter_RGB(save_path + "//RGB//"+"ddd_RGB_" + subjectnumber + "_" + classnumber + ".avi", CV_FOURCC('M','J','P','G'), 30, Size(320,240), true);
	VideoWriter VideoWriter_DEP(save_path + "//DEP//"+"ddd_DEP_" + subjectnumber + "_" + classnumber + ".avi", CV_FOURCC('M','J','P','G'), 30, Size(320,240), true); 
	//VideoWriter VideoWriter_RGB(save_path+"/"+"Class1"+"/"+"RGB"+"/"+"ddd_train_class1_1_RGB.avi", CV_FOURCC('M','J','P','G'), 30, Size(640,480), true);  
	//VideoWriter VideoWriter_DEP(save_path+"/"+"Class1"+"/"+"DEP"+"/"+"ddd_train_class1_1_DEP.avi", CV_FOURCC('M','J','P','G'), 30, Size(640,480), true); 
#endif

/*----------------------------------------------------------------------------*/
// New audio sample event handler
void onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data)
{
    //printf("A#%u: %d\n",g_aFrames,data.audioData.size());
    g_aFrames++;
}

/*----------------------------------------------------------------------------*/
// New color sample event handler
void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data)
{
    //printf("C#%u: %d\n",g_cFrames,data.colorMap.size());
    g_cFrames++;

	//ddd
	//Generate the color map
	Mat imageColor( 480, 640, CV_8UC3, (void*)(const uint8_t*)data.colorMap );
	//imshow("imageColor",imageColor);
	Mat imageColorResize;
	resize(imageColor,imageColorResize,Size(320,240),0,0,INTER_CUBIC);
	imshow("imageColorResize",imageColorResize);
	
	imageColorResizePub=imageColorResize;
	//imshow("imageColorResizePub",imageColorResizePub);

	//record
#ifdef RECORD
	VideoWriter_RGB.write( imageColorResize );
#endif

	waitKey(1);
}

/*----------------------------------------------------------------------------*/
// New depth sample event handler
void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
    //printf("Z#%u: %d\n",g_dFrames,data.vertices.size());
	//print the size of the vertices = 320*240 = 76800, vertice represents the cartesian 3D coordinates of each pixel,
	//expressed in millimeters. Saturated pixels are given the special value 32002

    // Project some 3D points in the Color Frame
    if (!g_pProjHelper)
    {
        g_pProjHelper = new ProjectionHelper (data.stereoCameraParameters);
        g_scp = data.stereoCameraParameters;
    }
    else if (g_scp != data.stereoCameraParameters)
    {
        g_pProjHelper->setStereoCameraParameters(data.stereoCameraParameters);
        g_scp = data.stereoCameraParameters;
    }

    int32_t w, h;
    FrameFormat_toResolution(data.captureConfiguration.frameFormat,&w,&h); 
	//Retrieve the width(w) and height(h) of the format. (in pixel)
    int cx = w/2;	
    int cy = h/2;
	//printf("%d %d",cx,cy);

    Vertex p3DPoints[4];
	//The Vertex struct holds the position of a point in space as defined by its 3D integer coordinates

	//Only project 4 points to color frames
    p3DPoints[0] = data.vertices[(cy-h/4)*w+cx-w/4];
    p3DPoints[1] = data.vertices[(cy-h/4)*w+cx+w/4];
    p3DPoints[2] = data.vertices[(cy+h/4)*w+cx+w/4];
    p3DPoints[3] = data.vertices[(cy+h/4)*w+cx-w/4];
    
    Point2D p2DPoints[4];
    g_pProjHelper->get2DCoordinates ( p3DPoints, p2DPoints, 4, CAMERA_PLANE_COLOR);

    g_dFrames++;

	//ddd
	// Generate the depth map
	Mat imageDepth( h, w, CV_16UC1, (void*)(const int16_t*)data.depthMap );
	imshow("imageDepth",imageDepth);

	Mat imageDepthScaled,imageDepthScaledRGB;
	double iMinDepth,iMaxDepth;
	const int depthThreshold_depth = 1000;
	minMaxIdx(imageDepth,&iMinDepth,&iMaxDepth);
	//printf("%f",iMaxDepth);
	imageDepth.convertTo(imageDepthScaled, CV_8U, (uint8_t)30*255.0/iMaxDepth); //iMaxDepth is about 32000, too large, so *30
	imshow("imageDespthScaled",imageDepthScaled);
	cvtColor( imageDepthScaled, imageDepthScaledRGB, CV_GRAY2RGB );
	imshow("imageDespthScaledRGB",imageDepthScaledRGB);

	//record
#ifdef RECORD
	VideoWriter_DEP.write( imageDepthScaledRGB );
#endif

	//Mat imageDepthConfidence( h, w, CV_16UC1, (void*)(const int16_t*)data.confidenceMap );
	//imshow("imageDepthConfidence",imageDepthConfidence);

    // Quit the main loop after 200 depth frames received
    //if (g_dFrames == 200)
    //    g_context.quit();
	if(waitKey(1) == 'q')
        g_context.quit(); //stops run()

}

/*----------------------------------------------------------------------------*/
void configureAudioNode()
{
    g_anode.newSampleReceivedEvent().connect(&onNewAudioSample);

    AudioNode::Configuration config = g_anode.getConfiguration();
    config.sampleRate = 44100;

    try 
    {
        g_context.requestControl(g_anode,0);

        g_anode.setConfiguration(config);
        
        g_anode.setInputMixerLevel(0.5f);
    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }
}

/*----------------------------------------------------------------------------*/
void configureColorNode()
{
    // connect new color sample handler
    g_cnode.newSampleReceivedEvent().connect(&onNewColorSample);

    ColorNode::Configuration config = g_cnode.getConfiguration();
    config.frameFormat = FRAME_FORMAT_VGA;
    config.compression = COMPRESSION_TYPE_MJPEG;
    config.powerLineFrequency = POWER_LINE_FREQUENCY_50HZ;
    //config.framerate = 25;
	config.framerate = 30;

    g_cnode.setEnableColorMap(true);

    try 
    {
        g_context.requestControl(g_cnode,0);

        g_cnode.setConfiguration(config);
    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (IOException& e)
    {
        printf("IO Exception: %s\n",e.what());
    }
    catch (InvalidOperationException& e)
    {
        printf("Invalid Operation Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }
}

/*----------------------------------------------------------------------------*/
void configureDepthNode()
{
    g_dnode.newSampleReceivedEvent().connect(&onNewDepthSample);

    DepthNode::Configuration config = g_dnode.getConfiguration();
    config.frameFormat = FRAME_FORMAT_QVGA;
    //config.framerate = 25;
	config.framerate = 30;
    config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE; //CAMERA_MODE_LONG_RANGE or CAMERA_MODE_CLOSE_MODE
    config.saturation = true;

    g_dnode.setEnableVertices(true);
	//ddd
	g_dnode.setEnableDepthMap(true);
	g_dnode.setEnableConfidenceMap(true);
	//g_dnode.setEnableDepthMapFloatingPoint(true);

    try 
    {
        g_context.requestControl(g_dnode,0);
		//g_context request full control on the node. 
		//Only one client (i.e. one context) at a time can modify the configuration of a node.
        g_dnode.setConfiguration(config);
    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (IOException& e)
    {
        printf("IO Exception: %s\n",e.what());
    }
    catch (InvalidOperationException& e)
    {
        printf("Invalid Operation Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }

}

/*----------------------------------------------------------------------------*/
void configureNode(Node node)
{
    if ((node.is<DepthNode>())&&(!g_dnode.isSet()))
    {
        g_dnode = node.as<DepthNode>();
        configureDepthNode();
        g_context.registerNode(node);
    }

    if ((node.is<ColorNode>())&&(!g_cnode.isSet()))
    {
        g_cnode = node.as<ColorNode>();
        configureColorNode();
        g_context.registerNode(node);
    }

    if ((node.is<AudioNode>())&&(!g_anode.isSet()))
    {
        g_anode = node.as<AudioNode>();
        configureAudioNode();
        g_context.registerNode(node);
    }
}

/*----------------------------------------------------------------------------*/
void onNodeConnected(Device device, Device::NodeAddedData data)
{
    configureNode(data.node);
}

/*----------------------------------------------------------------------------*/
void onNodeDisconnected(Device device, Device::NodeRemovedData data)
{
    if (data.node.is<AudioNode>() && (data.node.as<AudioNode>() == g_anode))
        g_anode.unset();
    if (data.node.is<ColorNode>() && (data.node.as<ColorNode>() == g_cnode))
        g_cnode.unset();
    if (data.node.is<DepthNode>() && (data.node.as<DepthNode>() == g_dnode))
        g_dnode.unset();
    printf("Node disconnected\n");
}

/*----------------------------------------------------------------------------*/
void onDeviceConnected(Context context, Context::DeviceAddedData data)
{
    if (!g_bDeviceFound)
    {
        data.device.nodeAddedEvent().connect(&onNodeConnected);
        data.device.nodeRemovedEvent().connect(&onNodeDisconnected);
        g_bDeviceFound = true;
    }
}

/*----------------------------------------------------------------------------*/
void onDeviceDisconnected(Context context, Context::DeviceRemovedData data)
{
    g_bDeviceFound = false;
    printf("Device disconnected\n");
}

int main(int argc, char* argv[])
{
    g_context = Context::create("localhost_ddd");

    g_context.deviceAddedEvent().connect(&onDeviceConnected);
    g_context.deviceRemovedEvent().connect(&onDeviceDisconnected);

    // Get the list of currently connected devices
    vector<Device> da = g_context.getDevices();

    // We are only interested in the first device
    if (da.size() >= 1)
    {
        g_bDeviceFound = true;

		//Connect the node adding and removing event to the device (Not sure about the mechanism)
        da[0].nodeAddedEvent().connect(&onNodeConnected);
        da[0].nodeRemovedEvent().connect(&onNodeDisconnected);

        vector<Node> na = da[0].getNodes(); //get the node array (Audio, Depth, Color)
        
        //printf("Found %u nodes\n",na.size());
        
        for (int n = 0; n < (int)na.size();n++)
            configureNode(na[n]); //Node Configuration; nodes are handled differently in the function according to their type
    }

	// create OpenCV Window
	//cv::namedWindow( "imageColor",  CV_WINDOW_AUTOSIZE );
	//cv::namedWindow( "imageColorResize",  CV_WINDOW_AUTOSIZE );
	//cv::namedWindow( "imageDepth",  CV_WINDOW_AUTOSIZE );
	//cv::namedWindow( "imageDespthScaled",  CV_WINDOW_AUTOSIZE );

    g_context.startNodes(); //Starts the capture on the registered nodes.

    g_context.run();

    g_context.stopNodes();

    if (g_cnode.isSet()) g_context.unregisterNode(g_cnode);
    if (g_dnode.isSet()) g_context.unregisterNode(g_dnode);
    if (g_anode.isSet()) g_context.unregisterNode(g_anode);

    if (g_pProjHelper)
        delete g_pProjHelper;

    return 0;
}
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <map>
#include <list>
#include <ctype.h>
#include <thread>        
#include <mutex>          

using namespace cv;
using namespace std;

#define MAXTIME 512
#define MAXSONDE 5

Mat frame;
mutex mtxFrame;
mutex mtxProbe;
mutex mtxSignal;
int nbSonde=MAXSONDE;
int sommeFPS=0;
int nbFPS=0;
int nbPointMax=1024; // Nombre de points maximum pour une sonde valeur par défaut 1024
int echRecou=16;
int nbSpectre=10; // Nombre de spectres intégrés pour la moyenne
int cptEchRecouv;
int indFFT=0;
int fps=0;
int freqSelec=-1;
int amplSelec=-1;
double Te=-10;
vector<Rect> probe; // zone étude si coordonnée x est égal -1 le point n'est pas utilisé
vector<int> debFile; // Pointeur sur le début des files
vector<vector<Vec4f> >probeValue;  // Tableau à taille fixe avec pointeur sur début pour simuler une file
vector<vector<Mat> > signal;       
Point lastPoint;
int indFrame=0;
double tpsFrame=0;
char mode = 's';
vector<Scalar> color={Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(255,255,255)};


int     fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
double  fontScale = 0.5;
int     thickness = 3;
int     baseline=0;



void help(char** av) {
    cout << "The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer." << endl
            << "Usage:\n" << av[0] << " <video file, image sequence or device number>" << endl
            << "q,Q,esc -- quit" << endl
            << "space   -- save frame" << endl << endl
            << "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*" << endl
            << "\texample: " << av[0] << " 0" << endl
            << "\tYou may also pass a video file instead of a device number" << endl
            << "\texample: " << av[0] << " video.avi" << endl
            << "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video." << endl
            << "\texample: " << av[0] << " right%%02d.jpg" << endl;
}


static void onMouse( int event, int x, int y, int, void* )
{
    if( event != EVENT_LBUTTONDOWN )
        return;

    lastPoint = Point(x,y);
}



void videoacquire(string s,Mat frame) 
{
    double		aaButter[11],bbButter[11];
    double x;
    int lastKey=-1;
    aaButter[0]=-0.9996859;
    aaButter[1]=-0.9993719;
    aaButter[2]=-0.9968633;
    aaButter[3]=-0.9937365;
    aaButter[4]=-0.9690674;
    aaButter[5]=-0.9390625;
    aaButter[6]=-0.7265425;
    aaButter[7]=-0.5095254;
    aaButter[8]=-0.3249;
    aaButter[9]=-0.1584;
    aaButter[10]=-0.0;
    bbButter[0]=0.0001571;
    bbButter[1]=0.0003141;
    bbButter[2]=0.0015683;
    bbButter[3]=0.0031318;
    bbButter[4]=0.0154663;
    bbButter[5]=0.0304687;
    bbButter[6]=0.1367287;
    bbButter[7]=0.2452373;
    bbButter[8]=0.3375;
    bbButter[9]=0.4208;
    bbButter[10]=0.5;
    string window_name = "video | q or esc to quit";
    cout << "press space to save a picture. q or esc to quit" << endl;
    namedWindow(window_name, WINDOW_AUTOSIZE); //resizable window;
    VideoCapture capture;
    capture.open(0);
    bool meanMode=false;
    int indMeanFilter=0;
	Mat     frame1;
	Mat     frame2;
    Mat    frameBuffer;
	std::vector<cv::Point2f> repereIni, repere;
	while (!capture.retrieve(frameBuffer));
	while (!capture.retrieve(frame2));
	while (!capture.retrieve(frame1));
    setMouseCallback( window_name, onMouse, 0 );
    double tpsInit = static_cast<double>(getTickCount());
    double tpsFramePre;
    tpsFrame=0;
    for (;;) {
        mtxFrame.lock();
        capture >> frame;
        fps++;
        indFrame++;
        tpsFramePre=tpsFrame;
        tpsFrame = (getTickCount() - tpsInit) / getTickFrequency();
        if (frame.empty())
            break;
		if (meanMode)	// Filtrage Butterworth
		{
            if (frame.size() != frame1.size())
                frame.copyTo(frame1);
            if (frame.size() != frame2.size())
                frame.copyTo(frame2);
			for (int i = 0; i<frame.rows; i++)
			{
				unsigned char *val = frame.ptr(i);
				unsigned char *valB1 = frame1.ptr(i);
				unsigned char *valB2 = frame2.ptr(i);
				for (int j = 0; j<frame.cols; j++)
					for (int k = 0; k<frame.channels(); k++, valB1++, valB2++, val++)
						*val = static_cast<uchar>(bbButter[indMeanFilter] * (*valB1 + *valB2) - aaButter[indMeanFilter] * *val);

			}
            swap(frame2,frame1);
            swap(frame,frame1);
		}
        mtxFrame.unlock();
        frame.copyTo(frameBuffer);
        mtxProbe.lock();
        for (int i=0; i<probe.size(); i++)
        {
            if (probe[i].x >= 0)
            {
                if (probe[i].x>=0 && probe[i].area()==1)
                    circle(frameBuffer, probe[i].br(), 5, color[i]);
                else 
                    rectangle(frameBuffer, probe[i], color[i]);
                putText(frameBuffer, format("%d",i),probe[i].br(),1,1,color[i]);

                if (mode=='r')
                {
                    Scalar v;
                    int indPre=debFile[i];
                    if (tpsFramePre > 0)
                    {
                        double pente = Te / (tpsFrame-tpsFramePre);
                        debFile[i]++;
                        if (debFile[i]>=nbPointMax)
                            debFile[i]=0;
                        if (probe[i].area()==1)
                            v=frame.at<Vec3b>(probe[i].br());
                        else
                            v= mean(frame(probe[i]));
                        Vec4f i1=probeValue[i][indPre],i2=Vec4f(tpsFrame,v[0],v[1],v[2]),ite;
                        ite = (i2-i1)*pente+i1;
                        probeValue[i][debFile[i]]=Vec4f(tpsFramePre+Te,min(255.F,max(ite[1],0.F)),min(255.F,max(ite[2],0.F)),min(255.F,max(ite[3],0.F)));
                    }
                    else
                    {
                        debFile[i]++;
                        if (debFile[i]>=nbPointMax)
                            debFile[i]=0;
                        if (probe[i].area()==1)
                            v=frame.at<Vec3b>(probe[i].br());
                        else
                            v= mean(frame(probe[i]));
                        probeValue[i][debFile[i]]=Vec4f(tpsFrame+Te,v[0],v[1],v[2]);
                   }
                }
            }

        }
        mtxProbe.unlock();
        imshow(window_name, frameBuffer);
        char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

        switch (key) {
        case '+':
            if (meanMode)
            {
               indMeanFilter++;
                if (indMeanFilter>10)
                    indMeanFilter=10;
            }
            else if (lastKey >= 0 && lastKey <= 4)
            {
                int r=probe[lastKey].width;
                if (r < 10)
                {
                    probe[lastKey].width++;
                    probe[lastKey].height++;
                }
            }
           break;
        case '-':
            if (meanMode)
            {
               indMeanFilter--;
               if (indMeanFilter<0)
                   indMeanFilter=10;
            }
            else if (lastKey >= 0 && lastKey <= 4)
            {
                int r=probe[lastKey-48].width;
                if (r>1)
                {
                    probe[lastKey-48].width--;
                    probe[lastKey-48].height--;
                }
           }
           break;
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
            mtxProbe.lock();
            if (mode == 'r')
            {
                mtxProbe.unlock();
                break;
            }
            probe[key-48]=Rect(lastPoint.x,lastPoint.y,1,1);
            lastKey=key - 48;
            mtxProbe.unlock();
            break;
        case 'm':
            meanMode=!meanMode;
            lastKey=-1;
            break;
        case 'q':
        case 'Q':
        case 27: //escape key
            mode =27;
            {
                ofstream fs("sonde.txt");
                for (int k=0;k<probe.size();k++)
                {
                    if (probe[k].x>=0)
                    for (int i = 0; i < probeValue[k].size(); i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            fs<<probeValue[k][j];
                            if (j!=3)
                                fs<<"\t";
                        }
                        fs << "\n";
                    }
                }
            }

            return ;
        case 'r':
            mode = 'r';
            lastKey=-1;
            break;
        case 's' :
            mode = 's';
            lastKey=-1;
            break;
        case 'g':
            x = capture.get(CAP_PROP_GAIN)-1;
            capture.set(CAP_PROP_GAIN,x);
            lastKey=-1;
            break;
        case 'G':
            x = capture.get(CAP_PROP_GAIN)+1;
            capture.set(CAP_PROP_GAIN,x);
            lastKey=-1;
            break;
        case 'b':
            x = capture.get(CAP_PROP_BRIGHTNESS)-1;
            capture.set(CAP_PROP_BRIGHTNESS,x);
            lastKey=-1;
            break;
        case 'B':
            x = capture.get(CAP_PROP_BRIGHTNESS)+1;
            capture.set(CAP_PROP_BRIGHTNESS,x);
            lastKey=-1;
            break;
        case 'E':
            x = capture.get(CAP_PROP_EXPOSURE)+1;
            capture.set(CAP_PROP_EXPOSURE,x);
            lastKey=-1;
            break;
        case 'e':
            x = capture.get(CAP_PROP_EXPOSURE)-1;
            capture.set(CAP_PROP_EXPOSURE,x);
            lastKey=-1;
            break;
        case 'c':
            x = capture.get(CAP_PROP_SATURATION)-1;
            capture.set(CAP_PROP_SATURATION,x);
            lastKey=-1;
            break;
        case 'C':
            x = capture.get(CAP_PROP_SATURATION)+1;
            capture.set(CAP_PROP_SATURATION,x);
            lastKey=-1;
            break;


            break;
        default:
            break;
        }
    }
    return ;
}


void TimeSignal()
{
    Mat red=Mat::zeros(300,1.2*MAXTIME,CV_8UC3);
    Mat green=Mat::zeros(300,1.2*MAXTIME,CV_8UC3);
    Mat blue=Mat::zeros(300,1.2*MAXTIME,CV_8UC3);
    imshow("Red componnent",red);
    imshow("Green componnent",green);
    imshow("Blue componnent",blue);
    vector<bool> sondeActive;
    waitKey(10);
    double fAcq=2/Te;
    std::this_thread::sleep_for (std::chrono::seconds(1));
    int indFramePre=-1;
    cptEchRecouv=0;
    sondeActive.resize(MAXSONDE);
    for (int i = 0; i<sondeActive.size();i++)
        sondeActive[i]=true;
    while (true)
    {
        mtxFrame.lock();
        if (mode == 27)
        {
            mtxFrame.unlock();
            return;
        }
        if (mode == 's')
        {
             
            mtxFrame.unlock();
            waitKey(30);

        }
        else
        {
            cptEchRecouv++;
            if (indFrame != indFramePre)
            {
                indFramePre = indFrame;
                mtxProbe.lock();
                if (probeValue.size() != 0)
                {
                    int minColor[4]={255,255,255,255};//{0,0,0,0};
                    int maxColor[4]={0,0,0,0};//{255,255,255,255};
                    int offsetRows=20;
                    for (int c = 0; c<probe.size(); c++)
                    {
                        if (probe[c].x >= 0 && probeValue[c][debFile[c]][0]!=0 && sondeActive[c])
                        {
                            for (int i = 0; i < probeValue[c].size(); i++)
                            {
                               if (probeValue[c][i][0] != 0)
                                {
                                    for (int j = 1; j < 4; j++)
                                    {
                                        if (probeValue[c][i][j]<minColor[j])
                                            minColor[j]=probeValue[c][i][j];
                                        if (probeValue[c][i][j]>maxColor[j])
                                            maxColor[j]=probeValue[c][i][j];
                                    }
                                }
                            }
                        }
                    }
                    if (cptEchRecouv == echRecou)
                        mtxSignal.lock();
                    fAcq=MAXTIME/(nbPointMax*Te);
                    for (int c=0; c<probeValue.size(); c++)
                    {
                        if (probe[c].x >= 0 && probeValue[c][debFile[c]][0]!=0 && sondeActive[c])
                        {
                            int fin = debFile[c]+1;
                            if (fin>=nbPointMax)
                                fin=0;
                            while (probeValue[c][fin][0] == 0)
                            {
                                fin++;
                                if (fin>=nbPointMax)
                                    fin=0;
                            }
                            float tpsIni = probeValue[c][fin][0];
                            int ind1 = fin,ind2=fin+1;
                            if (ind1>=nbPointMax)
                                ind1=0;
                            if (ind2>=nbPointMax)
                                ind2=ind2-nbPointMax;
                            for (int i = 1; i < probeValue[c].size(); i++)
                            {
                                if (probeValue[c][ind1][0] != 0 && probeValue[c][ind2][0] && probeValue[c][ind1][0]<probeValue[c][ind2][0])
                                {
                                    double x1=probeValue[c][ind1][0]-tpsIni;
                                    double x2=probeValue[c][ind2][0]-tpsIni;
                                    line(green,Point(x1*fAcq, 255-(probeValue[c][ind1][2]-minColor[2])/(maxColor[2]-minColor[2]+1)*255+offsetRows),Point(x2*fAcq, 255-(probeValue[c][ind2][2]-minColor[2])/(maxColor[2]-minColor[2]+1)*255+offsetRows),color[c],1);
                                    line(red,Point(x1*fAcq, 255-(probeValue[c][ind1][1]-minColor[1])/(maxColor[1]-minColor[1]+1)*255+offsetRows),Point(x2*fAcq,255- (probeValue[c][ind2][1]-minColor[1])/(maxColor[1]-minColor[1]+1)*255+offsetRows),color[c],1);
                                    line(blue,Point(x1*fAcq, 255-(probeValue[c][ind1][3]-minColor[3])/(maxColor[3]-minColor[3]+1)*255+offsetRows),Point(x2*fAcq, 255-(probeValue[c][ind2][3]-minColor[3])/(maxColor[3]-minColor[3]+1)*255+offsetRows),color[c],1);
                                }
                                ind1=ind2;
                                ind2++;
                                if (ind2>=nbPointMax)
                                    ind2=0;
                            }
                        }
                    }
                    if (cptEchRecouv == echRecou)
                    {
                        cptEchRecouv=0;
                        for (int c=0; c<probeValue.size(); c++)
                        {
                            if (probe[c].x >= 0 && probeValue[c][debFile[c]][0] != 0 && signal[c][0].cols== probeValue[c].size())
                            {
                                indFFT++;
                                float *ptr = (float*)signal[c][0].ptr();
                                for (int i = 0; i < probeValue[c].size() ; i++,ptr++)
                                    *ptr = probeValue[c][i][1];
                                ptr = (float*)signal[c][1].ptr();
                                for (int i = 0; i < probeValue[c].size() ; i++,ptr++)
                                    *ptr = probeValue[c][i][2];      
                                ptr = (float*)signal[c][2].ptr();
                                for (int i = 0; i < probeValue[c].size() ; i++,ptr++)
                                    *ptr = probeValue[c][i][3];
                            }
                        }
                        mtxSignal.unlock();
                    }

                }
                mtxProbe.unlock();
                imshow("Red componnent",red);
                imshow("Green componnent",green);
                imshow("Blue componnent",blue);
                red=Mat::zeros(300,1.2*MAXTIME,CV_8UC3);
                green=Mat::zeros(300,1.2*MAXTIME,CV_8UC3);
                blue=Mat::zeros(300,1.2*MAXTIME,CV_8UC3);
            }
            else if (cptEchRecouv == echRecou)
                cptEchRecouv=0;
            mtxFrame.unlock();
        }
        char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

        switch (key) {
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
            sondeActive[key-48] = !sondeActive[key-48];
            break;
        case 'a':
        {
            for (int i = 0; i<sondeActive.size();i++)
                sondeActive[i]= true;

        }
        default:
            break;
        }




    }
}

void onSelecFreq(int event, int x, int y, int, void*)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        freqSelec=x;
        amplSelec=y;
    }
    if (event == EVENT_RBUTTONDOWN)
    {
        freqSelec=-1;
        amplSelec=-1;
    }
}

#define CURVE_ROWS 400
void SpectreSignal()
{
    Mat red=Mat::zeros(CURVE_ROWS,2*MAXTIME,CV_8UC3);
    Mat green=Mat::zeros(CURVE_ROWS,2*MAXTIME,CV_8UC3);
    Mat blue=Mat::zeros(CURVE_ROWS,2*MAXTIME,CV_8UC3);
    imshow("FFT Red componnent",red);
    imshow("FFT Green componnent",green);
    imshow("FFT Blue componnent",blue);
    setMouseCallback( "FFT Red componnent", onSelecFreq, 0 );
    setMouseCallback( "FFT Green componnent", onSelecFreq, 0 );
    setMouseCallback( "FFT Blue componnent", onSelecFreq, 0 );
    vector<bool> sondeActive;
    sondeActive.resize(MAXSONDE);
    for (int i = 0; i<sondeActive.size();i++)
        sondeActive[i]=true;
    waitKey(10);
    double fAcq=10;
    std::this_thread::sleep_for (std::chrono::seconds(1));
    int indFFTPre=0;
    int nbFFT=0;
    vector<vector<Mat> > spectre;
    vector<vector<vector<Mat> >> serieSpectre; // liste des spectres successifs pour chaque sonde et chaque composante composante BGR
    spectre.resize(nbSonde);
    signal.resize(nbSonde);
    serieSpectre.resize(nbSpectre);
    for (int i=0;i<nbSpectre;i++)
    {
        serieSpectre[i].resize(nbSonde);
        for (int j = 0; j < nbSonde; j++)
        {
            serieSpectre[i][j].resize(3);
            for (int k=0;k<3;k++)
                serieSpectre[i][j][k] = Mat::zeros(1,nbPointMax,CV_32FC1);
        }
    }

    for (int i = 0; i < nbSonde; i++)
    {
        spectre[i].resize(3);
        signal[i].resize(3);
        for (int j = 0; j < 3; j++)
        {
            spectre[i][j] = Mat::zeros(1,nbPointMax,CV_32FC1);
            signal[i][j] = Mat::zeros(1,nbPointMax,CV_32FC1);
        }
    }
    int indSpectre=0;

    cptEchRecouv=0;
    while (true)
    {
        mtxFrame.lock();
        if (mode == 27)
        {
            mtxFrame.unlock();
            return;
        }
        mtxFrame.unlock();

        
        if (indFFT != indFFTPre)
        {
            vector<int> probeSelec;
            mtxFrame.lock();
            for (int c=0; c<probeValue.size(); c++)
            {
                if (probe[c].x >= 0 )
                    probeSelec.push_back(c);
            }
            mtxFrame.unlock();
            mtxSignal.lock();
            indFFTPre=indFFT;
            Mat s;
            for (int c=0; c<probeSelec.size(); c++)
            {
                vector<Mat> plan;
                for (int k = 0; k < 3; k++)
                {
                    spectre[probeSelec[c]][k] -= serieSpectre[indSpectre][probeSelec[c]][k];
                    dft(signal[probeSelec[c]][k],s,DFT_COMPLEX_OUTPUT);
                    split(s,plan);
                    magnitude(plan[0],plan[1],serieSpectre[indSpectre][probeSelec[c]][k]);
                    spectre[probeSelec[c]][k] += serieSpectre[indSpectre][probeSelec[c]][k];
                }
            }
            indSpectre++;
            if (indSpectre>=nbSpectre)
                indSpectre=0;
            int minColor[4]={255,255,255,255};//{0,0,0,0};
            int maxColor[4]={0,0,0,0};//{255,255,255,255};
            int offsetRows=20;
            int offsetCols=100;
            for (int c=0; c<probeSelec.size(); c++)
            {   
                float *ptr0 = (float*)spectre[probeSelec[c]][0].ptr(0);
                float *ptr1 = (float*)spectre[probeSelec[c]][1].ptr(0);
                float *ptr2 = (float*)spectre[probeSelec[c]][1].ptr(0);
                ptr0++;ptr1++,ptr2++;
                for (int i = 1; i < spectre[probeSelec[c]][0].cols/2; i++,ptr1++,ptr2++,ptr0++)
                {
                    if (sondeActive[c])
                    {
                        if (*ptr0<minColor[0])
                            minColor[0]=*ptr0;
                        if (*ptr0>maxColor[0])
                            maxColor[0]=*ptr0;
                        if (*ptr1<minColor[1])
                            minColor[1]=*ptr1;
                        if (*ptr1>maxColor[1])
                            maxColor[1]=*ptr1;
                        if (*ptr2<minColor[2])
                            minColor[0]=*ptr0;
                        if (*ptr2>maxColor[2])
                            maxColor[2]=*ptr2;
                    }
                }
            }
            for (int c=0; c<probeSelec.size(); c++)
            {
                float *ptr0 = (float*)spectre[probeSelec[c]][0].ptr(0);
                float *ptr1 = (float*)spectre[probeSelec[c]][1].ptr(0);
                float *ptr2 = (float*)spectre[probeSelec[c]][1].ptr(0);
                if (indFFT == 1024/echRecou)
                {
                    ofstream fs("spectre.txt");
                for (int i = 0; i < spectre[probeSelec[c]][0].cols; i++,ptr0++)
                    fs<<i<<"\t"<<signal[probeSelec[c]][0].at<float>(0,i)<<"\t"<<*ptr0<<endl;
                ptr0 = (float*)spectre[probeSelec[c]][0].ptr(0);
                }
                ptr0++;ptr1++;ptr2++;
                for (int i = 1; i < spectre[probeSelec[c]][0].cols/2; i++,ptr1++,ptr2++,ptr0++)
                {
                     if (sondeActive[c])
                    {
                        line(green,Point(i+offsetCols, 255+offsetRows),Point(i+offsetCols, 255-(*ptr0-minColor[0])/(maxColor[0]-minColor[0]+1)*255+offsetRows),color[probeSelec[c]],1);
                        line(red,Point(i+offsetCols, 255+offsetRows),Point(i+offsetCols,255- (*ptr1-minColor[1])/(maxColor[1]-minColor[1]+1)*255+offsetRows),color[probeSelec[c]],1);
                        line(blue,Point(i+offsetCols, 255+offsetRows),Point(i+offsetCols, 255-(*ptr2-minColor[2])/(maxColor[2]-minColor[2]+1)*255+offsetRows),color[probeSelec[c]],1);
                     }
                }

            }
            double ratioLegendx=1/5.0;
            double ratioLegendy=4/5.0;
            // X label
            for (int i = 0; i <= 5; i++)
            {
                    String s(format("%4.1f Hz",i/(10.0)/Te));
                    Size textSize = getTextSize(s, fontFace,fontScale, thickness, &baseline);           
                    putText(green, s,Point(i/5.0*spectre[probeSelec[0]][0].cols/2+offsetCols-textSize.width/2, 255+offsetRows+2*textSize.height),fontFace,fontScale,Scalar(255,255,255));
                    putText(red, s,Point(i/5.0*spectre[probeSelec[0]][0].cols/2+offsetCols-textSize.width/2, 255+offsetRows+2*textSize.height),fontFace,fontScale,Scalar(255,255,255));
                    putText(blue, s,Point(i/5.0*spectre[probeSelec[0]][0].cols/2+offsetCols-textSize.width/2, 255+offsetRows+2*textSize.height),fontFace,fontScale,Scalar(255,255,255));
                    //line(curve,Point(CURVE_COLS*ratioLegendx, curve.rows*(1-i/3.0)*ratioLegendy),Point(CURVE_COLS,curve.rows*(1-i/3.0)*ratioLegendy),Scalar(128,128,128),1);
            }
                    // Y axis
            line(green,Point(offsetCols,255+offsetRows),Point(offsetCols, offsetRows),Scalar(255,255,255),3);
            line(red,Point(offsetCols,255+offsetRows),Point(offsetCols, offsetRows),Scalar(255,255,255),3);
            line(blue,Point(offsetCols,255+offsetRows),Point(offsetCols, offsetRows),Scalar(255,255,255),3);
            // X axis
            line(red,Point(offsetCols, 255+offsetRows),Point(offsetCols+spectre[probeSelec[0]][0].cols/2,255+offsetRows),Scalar(255,255,255),3);
            line(green,Point(offsetCols, 255+offsetRows),Point(offsetCols+spectre[probeSelec[0]][0].cols/2,255+offsetRows),Scalar(255,255,255),3);
            line(blue,Point(offsetCols, 255+offsetRows),Point(offsetCols+spectre[probeSelec[0]][0].cols/2,255+offsetRows),Scalar(255,255,255),3);
            if (freqSelec != -1)
            {
                String s(format("%4.1f Hz",(freqSelec-offsetCols)/Te/spectre[probeSelec[0]][0].cols));
                putText(green, s,Point(offsetCols+spectre[probeSelec[0]][0].cols/2, 128),fontFace,fontScale,Scalar(255,255,255));
                putText(red, s,Point(offsetCols+spectre[probeSelec[0]][0].cols/2, 128),fontFace,fontScale,Scalar(255,255,255));
                putText(blue, s,Point(offsetCols+spectre[probeSelec[0]][0].cols/2, 128),fontFace,fontScale,Scalar(255,255,255));
           }


            mtxSignal.unlock();
            imshow("FFT Red componnent",red);
            imshow("FFT Green componnent",green);
            imshow("FFT Blue componnent",blue);
            waitKey(1);
            red=Mat::zeros(CURVE_ROWS,2*MAXTIME,CV_8UC3);
            green=Mat::zeros(CURVE_ROWS,2*MAXTIME,CV_8UC3);
            blue=Mat::zeros(CURVE_ROWS,2*MAXTIME,CV_8UC3);
       }
        else
        {
            std::this_thread::sleep_for (std::chrono::milliseconds(100));;
        }
        char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

        switch (key) {
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
            sondeActive[key-48] = !sondeActive[key-48];
            break;
        case 'a':
        {
            for (int i = 0; i<sondeActive.size();i++)
                sondeActive[i]= true;

        }
        default:
            break;
        }


    }
}




int main(int ac, char** av) 
{



  probe.resize(nbSonde);
  probeValue.resize(nbSonde);
  signal.resize(nbSonde);
  debFile.resize(nbSonde);
  for (int i = 0; i < probe.size(); i++)
  {
    probe[i] = Rect(-1,-1,0,0);
    probeValue[i].resize(nbPointMax);
    signal[i].resize(nbPointMax);
    debFile[0]=0;
    for (int j = 0; j < nbPointMax; j++)
    {
        probeValue[i][j] = Vec4f(0,0,0,0);
    }
  }
    string arg = "";
    thread thAcq(videoacquire,arg,frame);
    thread thSig(TimeSignal);
    thread thFFT(SpectreSignal);

    thAcq.detach();
    thSig.detach();
    thFFT.detach();

    while (true)
    {
        std::this_thread::sleep_for (std::chrono::seconds(1));
        mtxFrame.lock();
        if (mode==27)
            break;;
        sommeFPS+=fps;
        nbFPS++;
        cout << "FPS = "<<sommeFPS/nbFPS<<"("<<fps<<")\tTe="<<Te<<endl;
        fps=0;
        if (Te<0)
            Te++;
        else if (Te==0)
            Te=double(nbFPS)/sommeFPS;
        mtxFrame.unlock();
    }
    std::this_thread::sleep_for (std::chrono::seconds(1));
    return 0;
}

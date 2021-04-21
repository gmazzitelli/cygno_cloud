#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include <stdio.h>
#include <iostream>
#include "TGraph.h"
#include "TFile.h"
#include "TLine.h"
#include <TF1.h>
#include <TVector.h>
#include "TDirectory.h"

//101 thr
//binSize 4

void clus1(int run = 3958, int img = 0, float thr = 101, float binSize = 4){
  gDirectory->DeleteAll();
  gStyle->SetPalette(0);
  FILE *dataC1;

  int row, col, x, y, x2, y2, i, j, k, u, v, w;
  int nRow = 2048;
  int nCol = 2048;
  int intBinSize = binSize;
  int nBinX = nRow/intBinSize;
  int nBinY = nCol/intBinSize;
  float light, pedestal;
  int found = 0;
  int minFound=999;
  int maxFound=-999;
  int foundArray[8];
  int clu, size, sizeBin;

  int cut = 125;

  char path[200];
  char tmp[200];

  TH2F *hImage          = new TH2F("hImage",         "",nCol,0,nCol,nRow,0,nRow);
  TH2F *hImageRaw       = new TH2F("hImageRaw",      "",nCol,0,nCol,nRow,0,nRow);
  TH2F *hImageBin       = new TH2F("hImageBin",      "",nBinX,0,nBinX,nBinY,0,nBinY);
  TH2F *hImageBinDig    = new TH2F("hImageBinDig",   "",nBinX,0,nBinX,nBinY,0,nBinY);
  TH2F *hImageBinClu    = new TH2F("hImageBinClu",   "",nBinX,0,nBinX,nBinY,0,nBinY);
  TH2F *hImagePed       = new TH2F("hImagePed",      "",nCol,0,nCol,nRow,0,nRow);

  TH1F *hall            = new TH1F("hall",        "",1000, 0, 1000);
  TH1F *hall_ped        = new TH1F("hall_ped",        "",1000, 0, 1000);

  TF1 *g = new TF1("g","gaus(x)");
  g->SetParameters(7e5,100,2);
  hImageRaw -> SetStats(kFALSE);
  hImageBin -> SetStats(kFALSE);
  hImageBinDig -> SetStats(kFALSE);
  hImageBinClu -> SetStats(kFALSE);

  //TFile *p = new TFile("/Users/davide/Documents/Cygnus/Lab_Oct_2018/Results/ped_run818.root","READ");

  char ped[1000];
//  TFile *p = TFile::Open("https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygnus/Data/LAB/histograms_Run03965.root");
//  sprintf(ped,"/home/student/CYGNO/histograms_Run03965.root");
//  TFile *p = new TFile(ped,"READ");
  TFile *p = new TFile("./ped_run3965.root","READ");
  if(p->IsOpen()) {
    //      cout<<" File "<<name<<" opened!"<<endl;
  } else {
    cout<<"Pedestal file does not exist!"<<endl;
    return;
  }
  //sprintf(ped,"pic_run03965_ev%d",img);
  hImagePed = (TH2F*)gDirectory->Get("hImagePed")->Clone();
  hImagePed->SetDirectory(0);
  p->Close();
  //hImagePed->Draw("colz");
  //  sprintf(path,"/Users/davide/Documents/Sbai/Mondo/AmBe_2018/Data/run%03d/img%04.0f.DAT", run, img);
  char name[1000], nameImg[1000];
  double value;
  //  sprintf(name,"/Users/davide/Documents/Sbai/Mondo/Lab_Oct_2018/Data/Run%03d/Data.root", run);  
  //sprintf(name,"/Users/davide/Documents/Cygnus/Lab_Oct_2018/Data/Run%03d/Data.root", run); 
  sprintf(name,"https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygnus/Data/LAB/histograms_Run%05d.root",run);
  TFile *f = TFile::Open(name);
    
  if (f!=NULL){
      cout<<" File "<<name<<" opened!"<<endl;
  } else {
      cout<<" File "<<name<<" does not exists!"<<endl;
      return;
  }
//  if(f->IsOpen()) {
//    cout<<" File "<<name<<" opened!"<<endl;
//  } else {
//    cout<<" File "<<name<<" does not exists!"<<endl;
//    return;
//  }
  
  //sprintf(nameImg,"run%03d_%04.0f", run, img); 
  sprintf(nameImg,"pic_run%05d_ev%d",run,img);

  hImageRaw = (TH2F*)gDirectory->Get(nameImg)->Clone();
  hImageRaw->SetDirectory(0);
  f->Close();

  for(row=1; row<=nRow; row++){
    for(col=1; col<=nCol; col++){
      pedestal=hImagePed->GetBinContent(col, row);
      light = hImageRaw->GetBinContent(col, row);
      hall->Fill(light);
      light = light - pedestal+99;
      hall_ped->Fill(light);
      if(light>cut) light = cut;
      if(light<90)  light = 90;
      if(pedestal<90||pedestal>110) light = 99;//90,110
      hImage   ->Fill(col        , row        , light);
      hImageBin->Fill(col/binSize, row/binSize, light/(binSize*binSize));
    }
  }
  
  
  for(y=1; y<nBinY; y++){
    for(x=1; x<nBinX; x++){
      light = hImageBin->GetBinContent(x, y);
      if(light>thr){ 
	hImageBinDig->SetBinContent(x, y, 1);
	hImageBinClu->SetBinContent(x, y, 1);
      }
      if(light<=thr){      
	hImageBinDig->SetBinContent(x, y, 0);
	hImageBinClu->SetBinContent(x, y, 0);
      }
    }
  }
  hImageBinClu->Draw("colz");
  clu = 1;
  
  for(y=1; y<nBinY-1; y++){
    for(x=1; x<nBinX-1; x++){
      value = hImageBinDig->GetBinContent(x, y);
      if(value){
	for(w=0; w<8; w++){
	  foundArray[w]=-1;
	}
	v=0;
	maxFound = -999;
	minFound =  999;
	found = 0;
	for(i=-1; i<=1; i++){
	  for(j=-1; j<=1; j++){
	    //found = 0; //modifica contro baco 
	    if(i||j) found = hImageBinClu->GetBinContent(x+i, y+j);
	    if(found){
	      if(found>maxFound) maxFound = found;
	      if(found<minFound) minFound = found;
	      for(w=0; w<8; w++){
		if(foundArray[w]==found) break;
	      }
	      foundArray[v]=found;
	      v++;
	    }
	  }
	}
 
	if(v){
	  if(maxFound==1){	
	    clu++;
	    for(i=-1; i<=1; i++){
	      for(j=-1; j<=1; j++){
		if(hImageBinDig->GetBinContent(x+i, y+j)) hImageBinClu->SetBinContent(x+i, y+j, clu);
	      }
	    }
	  }
	 
	  if(maxFound>1){	
	    for(i=-1; i<=1; i++){
	      for(j=-1; j<=1; j++){
		if(hImageBinDig->GetBinContent(x+i, y+j)){ 
		  hImageBinClu->SetBinContent(x+i, y+j, maxFound);
		  //		  printf("Point %d-%d Set to %d X: %d - Y: %d\n", x, y, maxFound, x+i, y+j);
		}
 	      }
	    }
	    for(y2=1; y2<=y; y2++){
	      for(x2=1; x2<=nBinX; x2++){
		value = hImageBinClu->GetBinContent(x2, y2);
		for(w=0; w<v; w++){
		  if((value!=1)&&(value==foundArray[w])) hImageBinClu->SetBinContent(x2, y2, maxFound);
		}
	      }
	    }
	  }
	}
      }
    }
  }
  //hImage->Draw();
  printf("Found %d clusters\n",clu);  
  //hImageBinClu->Draw("colz");
  char outname[100];
  TVector vclu(1);
  vclu[0]=clu;
  //sprintf(outname,"/Users/davide/Documents/Cygnus/Lab_Oct_2018/Results/clus1_run%03d_img%03.0f_thr%.1f_bin%.0f_ped.root", run, img, thr, binSize);
  sprintf(outname,"./clus1_run%05d_img%d_thr%.1f_bin%.0f.root", run, img, thr, binSize);
  TFile *outfile = new TFile(outname, "RECREATE");
  hall           ->Write();
  hall_ped       ->Write();
  hImage         ->Write();
  hImageBin      ->Write();
  hImageBinDig   ->Write();
  hImageBinClu   ->Write();
  vclu.Write("vclu")      ;
  outfile        ->Write();
  outfile        ->Close();
  

}

void scan1(int run, int start, int end = 99){
  for(int i=start; i<end; i++){
    printf("%d\n", i);
    clus1(run, i, 100.5);
  }
}

void scan(int run, float thr=100.5){
  for(int i=0; i<100; i++){
    printf("%d\n", i);
    clus1(run, i, thr);
  }
}

void scan1_run(int start = 0, int end = 100, float thr = 100.5){
  /*
  for(int run=848; run<861; run++){
    for(int i=start; i<end; i++){
      printf("run: %d  %d\n", run, i);
      clus1(run, i, 100);
    }
  }
  */
  for(int run=836; run<=841; run++){
    for(int i=start; i<end; i++){
      printf("run: %d  %d\n", run, i);
      clus1(run, i, thr);
    }
  }
}


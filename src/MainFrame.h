#pragma once
#include <wx/wx.h>
#include <vector>
#include "SiftDetector.h"

class MainFrame : public wxFrame {
public:
    MainFrame(const wxString& title);

private:
    void OnLoadImage1(wxCommandEvent& event);
    void OnLoadImage2(wxCommandEvent& event);
    void OnHarrisDetect(wxCommandEvent& event);
    void OnLambdaMinusDetect(wxCommandEvent& event);
    void OnSiftExtract(wxCommandEvent& event);
    void OnMatchImageSet(wxCommandEvent& event);
    void OnExit(wxCommandEvent& event);
    void OnPaint(wxPaintEvent& event);

    void UpdateDisplay(const wxImage& img);
    void EnsureDescriptorsReady();

    wxImage m_originalImage;
    wxImage m_secondImage;
    wxImage m_displayImage;
    wxImage m_wxDisplayImage;

    wxButton* m_btnLoad1;
    wxButton* m_btnLoad2;
    wxButton* m_btnHarris;
    wxButton* m_btnLambda;
    wxButton* m_btnSift;
    wxButton* m_btnMatch;
    wxStaticText* m_lblStatus;
    wxStaticText* m_lblTips;
    
    wxStaticText* m_lblHarrisThresh;
    wxSlider* m_sldHarrisThreshold;
    wxStaticText* m_lblLambdaThresh;
    wxSlider* m_sldLambdaThreshold;
    
    wxPanel* m_imagePanel;

    std::vector<SiftDescriptor> m_descImage1;
    std::vector<SiftDescriptor> m_descImage2;
    bool m_desc1Dirty = true;
    bool m_desc2Dirty = true;

    wxDECLARE_EVENT_TABLE();
};

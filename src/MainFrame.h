#pragma once
#include <wx/wx.h>
#include <vector>

class MainFrame : public wxFrame {
public:
    MainFrame(const wxString& title);

private:
    void OnLoadImage(wxCommandEvent& event);
    void OnHarrisDetect(wxCommandEvent& event);
    void OnLambdaMinusDetect(wxCommandEvent& event);
    void OnSiftExtract(wxCommandEvent& event);
    void OnExit(wxCommandEvent& event);
    void OnPaint(wxPaintEvent& event);

    void UpdateDisplay(const wxImage& img);

    wxImage m_originalImage;
    wxImage m_displayImage;
    wxImage m_wxDisplayImage;

    wxButton* m_btnLoad;
    wxButton* m_btnHarris;
    wxButton* m_btnLambda;
    wxButton* m_btnSift;
    wxStaticText* m_lblStatus;
    
    wxStaticText* m_lblHarrisThresh;
    wxSlider* m_sldHarrisThreshold;
    wxStaticText* m_lblLambdaThresh;
    wxSlider* m_sldLambdaThreshold;
    
    wxPanel* m_imagePanel;

    wxDECLARE_EVENT_TABLE();
};

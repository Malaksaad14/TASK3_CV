#include "MainFrame.h"
#include "HarrisDetector.h"
#include "SiftDetector.h"
#include <chrono>

wxBEGIN_EVENT_TABLE(MainFrame, wxFrame)
    EVT_BUTTON(1001, MainFrame::OnLoadImage)
    EVT_BUTTON(1002, MainFrame::OnHarrisDetect)
    EVT_BUTTON(1003, MainFrame::OnLambdaMinusDetect)
    EVT_BUTTON(1004, MainFrame::OnSiftExtract)
wxEND_EVENT_TABLE()

MainFrame::MainFrame(const wxString& title)
    : wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(1024, 768)) {
    
    wxBoxSizer* mainSizer = new wxBoxSizer(wxHORIZONTAL);
    
    // Left panel for controls
    wxPanel* controlPanel = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxSize(200, -1));
    wxBoxSizer* controlSizer = new wxBoxSizer(wxVERTICAL);
    
    m_btnLoad = new wxButton(controlPanel, 1001, "Load Image");
    m_btnHarris = new wxButton(controlPanel, 1002, "Harris Detect");
    m_btnLambda = new wxButton(controlPanel, 1003, "Lambda- Detect");
    m_btnSift = new wxButton(controlPanel, 1004, "SIFT Extract");
    m_lblStatus = new wxStaticText(controlPanel, wxID_ANY, "Time: N/A");
    
    controlSizer->Add(m_btnLoad, 0, wxALL | wxEXPAND, 5);
    controlSizer->Add(m_btnHarris, 0, wxALL | wxEXPAND, 5);
    controlSizer->Add(m_btnLambda, 0, wxALL | wxEXPAND, 5);
    controlSizer->Add(m_btnSift, 0, wxALL | wxEXPAND, 5);
    controlSizer->Add(m_lblStatus, 0, wxALL | wxEXPAND, 5);
    
    controlPanel->SetSizer(controlSizer);
    
    // Right panel for image viewing
    m_imagePanel = new wxPanel(this, wxID_ANY);
    m_imagePanel->SetBackgroundColour(wxColour(30, 30, 30));
    m_imagePanel->Bind(wxEVT_PAINT, &MainFrame::OnPaint, this);
    
    mainSizer->Add(controlPanel, 0, wxEXPAND, 0);
    mainSizer->Add(m_imagePanel, 1, wxEXPAND, 0);
    
    this->SetSizer(mainSizer);
    this->Center();
}

void MainFrame::OnLoadImage(wxCommandEvent& event) {
    wxFileDialog openFileDialog(this, _("Open Image file"), "", "",
                                "Image files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp",
                                wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    
    if (openFileDialog.ShowModal() == wxID_CANCEL)
        return;
        
    m_originalImage.LoadFile(openFileDialog.GetPath());
    if (m_originalImage.IsOk()) {
        m_displayImage = m_originalImage.Copy();
        UpdateDisplay(m_displayImage);
        m_lblStatus->SetLabel("Image loaded.");
    }
}

void MainFrame::OnHarrisDetect(wxCommandEvent& event) {
    if (!m_originalImage.IsOk()) return;
    
    m_lblStatus->SetLabel("Running Harris...");
    this->Update();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    int w = m_originalImage.GetWidth();
    int h = m_originalImage.GetHeight();
    
    // Convert wxImage RGB to MathUtils Grayscale natively
    MathUtils::Matrix2D gray = HarrisDetector::ConvertToGrayMatrix(m_originalImage.GetData(), w, h, 3);
    float k = 0.04f;
    float threshold = 5000000.0f; // Increased threshold to reduce weak corners
    std::vector<KeyPoint> keypoints = HarrisDetector::DetectHarris(gray, k, threshold, 0, 5); // NMS radius = 5 to separate points
    
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    m_displayImage = m_originalImage.Copy();
    
    // Draw points manually to wxImage via pixel manipulation
    unsigned char* data = m_displayImage.GetData();
    for (const auto& kp : keypoints) {
        // Simple 3x3 red circle
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int px = kp.x + dx;
                int py = kp.y + dy;
                if (px >= 0 && px < w && py >= 0 && py < h) {
                    int idx = (py * w + px) * 3;
                    data[idx] = 255;   // R
                    data[idx+1] = 0;   // G
                    data[idx+2] = 0;   // B
                }
            }
        }
    }
    
    UpdateDisplay(m_displayImage);
    m_lblStatus->SetLabel(wxString::Format("Harris: %d pts, %.1f ms", (int)keypoints.size(), ms));
}

void MainFrame::OnLambdaMinusDetect(wxCommandEvent& event) {
    if (!m_originalImage.IsOk()) return;
    
    m_lblStatus->SetLabel("Running Lambda-...");
    this->Update();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    int w = m_originalImage.GetWidth();
    int h = m_originalImage.GetHeight();
    
    MathUtils::Matrix2D gray = HarrisDetector::ConvertToGrayMatrix(m_originalImage.GetData(), w, h, 3);
    float threshold = 5000.0f; // Lowered significantly from 60000 - relies on NMS radius=5 to spread corners
    std::vector<KeyPoint> keypoints = HarrisDetector::DetectLambdaMinus(gray, threshold, 0, 5); // NMS radius = 5 to separate points
    
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    m_displayImage = m_originalImage.Copy();
    unsigned char* data = m_displayImage.GetData();
    for (const auto& kp : keypoints) {
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int px = kp.x + dx;
                int py = kp.y + dy;
                if (px >= 0 && px < w && py >= 0 && py < h) {
                    int idx = (py * w + px) * 3;
                    data[idx] = 0;     // R
                    data[idx+1] = 255; // G
                    data[idx+2] = 0;   // B
                }
            }
        }
    }
    
    UpdateDisplay(m_displayImage);
    m_lblStatus->SetLabel(wxString::Format("Lambda-: %d pts, %.1f ms", (int)keypoints.size(), ms));
}

void MainFrame::OnSiftExtract(wxCommandEvent& event) {
    if (!m_originalImage.IsOk()) return;
    
    m_lblStatus->SetLabel("Running SIFT Descriptor Extract...");
    this->Update();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    int w = m_originalImage.GetWidth();
    int h = m_originalImage.GetHeight();
    
    MathUtils::Matrix2D gray = HarrisDetector::ConvertToGrayMatrix(m_originalImage.GetData(), w, h, 3);
    std::vector<SiftDescriptor> descriptors = SiftDetector::ExtractFeatures(gray);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    m_displayImage = m_originalImage.Copy();
    unsigned char* data = m_displayImage.GetData();
    for (const auto& desc : descriptors) {
        // Draw blue cross
        int cx = desc.x;
        int cy = desc.y;
        for (int r = -3; r <= 3; r++) {
            if (cx + r >= 0 && cx + r < w) {
                int idx = (cy * w + cx + r) * 3;
                data[idx] = 0;     // R
                data[idx+1] = 0;   // G
                data[idx+2] = 255; // B
            }
            if (cy + r >= 0 && cy + r < h) {
                int idx = ((cy + r) * w + cx) * 3;
                data[idx] = 0;     // R
                data[idx+1] = 0;   // G
                data[idx+2] = 255; // B
            }
        }
    }
    
    UpdateDisplay(m_displayImage);
    m_lblStatus->SetLabel(wxString::Format("SIFT: %d desc, %.1f ms", (int)descriptors.size(), ms));
}

void MainFrame::OnExit(wxCommandEvent& event) {
    Close(true);
}

void MainFrame::UpdateDisplay(const wxImage& img) {
    if (!img.IsOk()) return;
    
    m_wxDisplayImage = img;
    m_imagePanel->Refresh();
}

void MainFrame::OnPaint(wxPaintEvent& event) {
    wxPaintDC dc(m_imagePanel);
    if (m_wxDisplayImage.IsOk()) {
        wxBitmap bmp(m_wxDisplayImage);
        
        // Scale to fit panel
        int pW, pH;
        m_imagePanel->GetSize(&pW, &pH);
        int iW = bmp.GetWidth(), iH = bmp.GetHeight();
        
        double scale = std::min((double)pW / iW, (double)pH / iH);
        int newW = (int)(iW * scale);
        int newH = (int)(iH * scale);
        
        wxImage scaledImg = m_wxDisplayImage.Scale(newW, newH, wxIMAGE_QUALITY_HIGH);
        wxBitmap scaledBmp(scaledImg);
        
        int x = (pW - newW) / 2;
        int y = (pH - newH) / 2;
        
        dc.DrawBitmap(scaledBmp, x, y, false);
    }
}

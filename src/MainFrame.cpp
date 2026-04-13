#include "MainFrame.h"
#include "HarrisDetector.h"
#include "SiftDetector.h"
#include <chrono>

wxBEGIN_EVENT_TABLE(MainFrame, wxFrame)
    EVT_BUTTON(1001, MainFrame::OnLoadImage1)
    EVT_BUTTON(1005, MainFrame::OnLoadImage2)
    EVT_BUTTON(1002, MainFrame::OnHarrisDetect)
    EVT_BUTTON(1003, MainFrame::OnLambdaMinusDetect)
    EVT_BUTTON(1004, MainFrame::OnSiftExtract)
    EVT_BUTTON(1006, MainFrame::OnMatchImageSet)
wxEND_EVENT_TABLE()

MainFrame::MainFrame(const wxString& title)
    : wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(1024, 768)) {
    
    wxBoxSizer* mainSizer = new wxBoxSizer(wxHORIZONTAL);
    
    // Left panel for controls
    wxPanel* controlPanel = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxSize(270, -1));
    controlPanel->SetBackgroundColour(wxColour(244, 248, 255));
    wxBoxSizer* controlSizer = new wxBoxSizer(wxVERTICAL);
    
    m_btnLoad1 = new wxButton(controlPanel, 1001, "Load Image 1");
    m_btnLoad2 = new wxButton(controlPanel, 1005, "Load Image 2");
    m_btnHarris = new wxButton(controlPanel, 1002, "Harris Detect");
    m_btnLambda = new wxButton(controlPanel, 1003, "Lambda- Detect");
    m_btnSift = new wxButton(controlPanel, 1004, "SIFT Extract");
    m_btnMatch = new wxButton(controlPanel, 1006, "Match Set (SSD + NCC)");
    m_lblStatus = new wxStaticText(controlPanel, wxID_ANY, "Ready");
    m_lblTips = new wxStaticText(controlPanel, wxID_ANY, "Load Image 1 + 2 first,\nthen run matching.");
    m_lblStatus->SetForegroundColour(wxColour(32, 44, 70));
    m_lblTips->SetForegroundColour(wxColour(75, 95, 130));
    wxFont statusFont = m_lblStatus->GetFont();
    statusFont.SetWeight(wxFONTWEIGHT_BOLD);
    m_lblStatus->SetFont(statusFont);
    
    // Sliders for thresholds
    m_lblHarrisThresh = new wxStaticText(controlPanel, wxID_ANY, "Harris Thresh: 5M");
    m_sldHarrisThreshold = new wxSlider(controlPanel, wxID_ANY, 50, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL);
    
    m_lblLambdaThresh = new wxStaticText(controlPanel, wxID_ANY, "Lambda Thresh: 5k");
    m_sldLambdaThreshold = new wxSlider(controlPanel, wxID_ANY, 50, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL);
    
    m_btnLoad1->SetBackgroundColour(wxColour(122, 181, 255));
    m_btnLoad1->SetForegroundColour(*wxWHITE);
    m_btnLoad2->SetBackgroundColour(wxColour(91, 157, 241));
    m_btnLoad2->SetForegroundColour(*wxWHITE);
    m_btnHarris->SetBackgroundColour(wxColour(110, 143, 252));
    m_btnHarris->SetForegroundColour(*wxWHITE);
    m_btnLambda->SetBackgroundColour(wxColour(79, 192, 143));
    m_btnLambda->SetForegroundColour(*wxWHITE);
    m_btnSift->SetBackgroundColour(wxColour(250, 165, 92));
    m_btnSift->SetForegroundColour(*wxWHITE);
    m_btnMatch->SetBackgroundColour(wxColour(171, 104, 245));
    m_btnMatch->SetForegroundColour(*wxWHITE);

    wxStaticBoxSizer* loadSizer = new wxStaticBoxSizer(wxVERTICAL, controlPanel, "Images");
    loadSizer->GetStaticBox()->SetForegroundColour(wxColour(38, 67, 125));
    loadSizer->Add(m_btnLoad1, 0, wxALL | wxEXPAND, 6);
    loadSizer->Add(m_btnLoad2, 0, wxLEFT | wxRIGHT | wxBOTTOM | wxEXPAND, 6);
    loadSizer->Add(m_lblTips, 0, wxLEFT | wxRIGHT | wxBOTTOM | wxEXPAND, 6);

    wxStaticBoxSizer* extractionSizer = new wxStaticBoxSizer(wxVERTICAL, controlPanel, "Feature Extraction");
    extractionSizer->GetStaticBox()->SetForegroundColour(wxColour(38, 67, 125));
    extractionSizer->Add(m_btnHarris, 0, wxALL | wxEXPAND, 6);
    extractionSizer->Add(m_lblHarrisThresh, 0, wxLEFT | wxRIGHT | wxTOP, 6);
    extractionSizer->Add(m_sldHarrisThreshold, 0, wxLEFT | wxRIGHT | wxBOTTOM | wxEXPAND, 6);
    extractionSizer->Add(m_btnLambda, 0, wxLEFT | wxRIGHT | wxBOTTOM | wxEXPAND, 6);
    extractionSizer->Add(m_lblLambdaThresh, 0, wxLEFT | wxRIGHT | wxTOP, 6);
    extractionSizer->Add(m_sldLambdaThreshold, 0, wxLEFT | wxRIGHT | wxBOTTOM | wxEXPAND, 6);
    extractionSizer->Add(m_btnSift, 0, wxLEFT | wxRIGHT | wxBOTTOM | wxEXPAND, 6);

    wxStaticBoxSizer* matchSizer = new wxStaticBoxSizer(wxVERTICAL, controlPanel, "Matching");
    matchSizer->GetStaticBox()->SetForegroundColour(wxColour(38, 67, 125));
    matchSizer->Add(m_btnMatch, 0, wxALL | wxEXPAND, 6);
    matchSizer->Add(m_lblStatus, 0, wxLEFT | wxRIGHT | wxBOTTOM | wxEXPAND, 6);

    controlSizer->Add(loadSizer, 0, wxALL | wxEXPAND, 6);
    controlSizer->Add(extractionSizer, 0, wxLEFT | wxRIGHT | wxBOTTOM | wxEXPAND, 6);
    controlSizer->Add(matchSizer, 0, wxLEFT | wxRIGHT | wxBOTTOM | wxEXPAND, 6);
    controlSizer->AddStretchSpacer();

    // Bind slider events
    m_sldHarrisThreshold->Bind(wxEVT_SLIDER, [this](wxCommandEvent& e) {
        float val = e.GetInt() * 100000.0f;
        m_lblHarrisThresh->SetLabel(wxString::Format("Harris Thresh: %.1fM", val / 1000000.0f));
    });
    m_sldLambdaThreshold->Bind(wxEVT_SLIDER, [this](wxCommandEvent& e) {
        float val = e.GetInt() * 100.0f;
        m_lblLambdaThresh->SetLabel(wxString::Format("Lambda Thresh: %.1fk", val / 1000.0f));
    });
    
    controlPanel->SetSizer(controlSizer);
    
    // Right panel for image viewing
    m_imagePanel = new wxPanel(this, wxID_ANY);
    m_imagePanel->SetBackgroundColour(wxColour(252, 254, 255));
    m_imagePanel->Bind(wxEVT_PAINT, &MainFrame::OnPaint, this);
    
    mainSizer->Add(controlPanel, 0, wxEXPAND, 0);
    mainSizer->Add(m_imagePanel, 1, wxEXPAND, 0);
    
    this->SetSizer(mainSizer);
    this->Center();
}

void MainFrame::OnLoadImage1(wxCommandEvent& event) {
    wxFileDialog openFileDialog(this, _("Open Image file"), "", "",
                                "Image files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp",
                                wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    
    if (openFileDialog.ShowModal() == wxID_CANCEL)
        return;
        
    m_originalImage.LoadFile(openFileDialog.GetPath());
    if (m_originalImage.IsOk()) {
        m_desc1Dirty = true;
        m_descImage1.clear();
        m_displayImage = m_originalImage.Copy();
        UpdateDisplay(m_displayImage);
        m_lblStatus->SetLabel("Image 1 loaded.");
    }
}

void MainFrame::OnLoadImage2(wxCommandEvent& event) {
    wxFileDialog openFileDialog(this, _("Open second image file"), "", "",
                                "Image files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp",
                                wxFD_OPEN | wxFD_FILE_MUST_EXIST);

    if (openFileDialog.ShowModal() == wxID_CANCEL)
        return;

    m_secondImage.LoadFile(openFileDialog.GetPath());
    if (m_secondImage.IsOk()) {
        m_desc2Dirty = true;
        m_descImage2.clear();
        m_imagePanel->Refresh();
        m_lblStatus->SetLabel("Image 2 loaded.");
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
    float threshold = m_sldHarrisThreshold->GetValue() * 100000.0f; 
    std::vector<KeyPoint> keypoints = HarrisDetector::DetectHarris(gray, k, threshold, 0, 5); 
    
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
    float threshold = m_sldLambdaThreshold->GetValue() * 100.0f;
    std::vector<KeyPoint> keypoints = HarrisDetector::DetectLambdaMinus(gray, threshold, 0, 5); 
    
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

void MainFrame::OnMatchImageSet(wxCommandEvent& event) {
    if (!m_originalImage.IsOk() || !m_secondImage.IsOk()) {
        m_lblStatus->SetLabel("Load image 1 and image 2 first.");
        return;
    }

    m_lblStatus->SetLabel("Preparing descriptors + matching...");
    this->Update();

    auto tFeat1 = std::chrono::high_resolution_clock::now();
    EnsureDescriptorsReady();
    auto tFeat2 = std::chrono::high_resolution_clock::now();

    auto tSSD1 = std::chrono::high_resolution_clock::now();
    std::vector<MatchPair> ssdMatches = SiftDetector::MatchDescriptorsSSD(m_descImage1, m_descImage2);
    auto tSSD2 = std::chrono::high_resolution_clock::now();

    auto tNCC1 = std::chrono::high_resolution_clock::now();
    std::vector<MatchPair> nccMatches = SiftDetector::MatchDescriptorsNCC(m_descImage1, m_descImage2);
    auto tNCC2 = std::chrono::high_resolution_clock::now();

    double featMs = std::chrono::duration<double, std::milli>(tFeat2 - tFeat1).count();
    double ssdMs = std::chrono::duration<double, std::milli>(tSSD2 - tSSD1).count();
    double nccMs = std::chrono::duration<double, std::milli>(tNCC2 - tNCC1).count();

    m_lblStatus->SetLabel(wxString::Format(
        "Set match | D1:%d D2:%d | SSD:%d (%.1f ms) | NCC:%d (%.1f ms) | Feat:%.1f ms",
        (int)m_descImage1.size(), (int)m_descImage2.size(),
        (int)ssdMatches.size(), ssdMs,
        (int)nccMatches.size(), nccMs,
        featMs
    ));
}

void MainFrame::EnsureDescriptorsReady() {
    if (m_desc1Dirty && m_originalImage.IsOk()) {
        int w1 = m_originalImage.GetWidth();
        int h1 = m_originalImage.GetHeight();
        MathUtils::Matrix2D gray1 = HarrisDetector::ConvertToGrayMatrix(m_originalImage.GetData(), w1, h1, 3);
        m_descImage1 = SiftDetector::ExtractFeatures(gray1);
        m_desc1Dirty = false;
    }

    if (m_desc2Dirty && m_secondImage.IsOk()) {
        int w2 = m_secondImage.GetWidth();
        int h2 = m_secondImage.GetHeight();
        MathUtils::Matrix2D gray2 = HarrisDetector::ConvertToGrayMatrix(m_secondImage.GetData(), w2, h2, 3);
        m_descImage2 = SiftDetector::ExtractFeatures(gray2);
        m_desc2Dirty = false;
    }
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
    if (m_originalImage.IsOk() && m_secondImage.IsOk()) {
        int pW, pH;
        m_imagePanel->GetSize(&pW, &pH);
        int gap = 16;
        int margin = 12;
        int cellW = (pW - (gap + 2 * margin));
        cellW = cellW / 2;
        int cellH = pH - 2 * margin;
        if (cellW < 1 || cellH < 1) return;

        wxImage leftSource = m_wxDisplayImage.IsOk() ? m_wxDisplayImage : m_originalImage;
        wxImage rightSource = m_secondImage;

        auto drawInCell = [&](const wxImage& source, int x0, const wxString& label, const wxColour& borderColor) {
            if (!source.IsOk()) return;
            double scale = std::min((double)cellW / source.GetWidth(), (double)cellH / source.GetHeight());
            int drawW = std::max(1, (int)(source.GetWidth() * scale));
            int drawH = std::max(1, (int)(source.GetHeight() * scale));
            wxImage scaled = source.Scale(drawW, drawH, wxIMAGE_QUALITY_HIGH);
            wxBitmap bmp(scaled);

            int x = x0 + (cellW - drawW) / 2;
            int y = margin + (cellH - drawH) / 2;

            dc.SetPen(wxPen(borderColor, 2));
            dc.SetBrush(*wxTRANSPARENT_BRUSH);
            dc.DrawRoundedRectangle(x0, margin, cellW, cellH, 6);
            dc.DrawBitmap(bmp, x, y, false);
            dc.SetTextForeground(borderColor);
            dc.DrawText(label, x0 + 8, margin + 6);
        };

        int leftX = margin;
        int rightX = margin + cellW + gap;
        drawInCell(leftSource, leftX, "Image 1", wxColour(66, 115, 210));
        drawInCell(rightSource, rightX, "Image 2", wxColour(95, 172, 124));
    } else if (m_wxDisplayImage.IsOk()) {
        wxBitmap bmp(m_wxDisplayImage);
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

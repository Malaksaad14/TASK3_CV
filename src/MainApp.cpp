#include "MainApp.h"
#include "MainFrame.h"

wxIMPLEMENT_APP(MainApp);

bool MainApp::OnInit() {
    wxInitAllImageHandlers();
    MainFrame* frame = new MainFrame("Feature Extraction (Harris & SIFT)");
    frame->Show(true);
    return true;
}

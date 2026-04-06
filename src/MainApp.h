#pragma once
#include <wx/wx.h>

class MainApp : public wxApp {
public:
    virtual bool OnInit();
};

wxDECLARE_APP(MainApp);

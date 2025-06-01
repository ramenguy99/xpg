#include <xpg/application.h>

using namespace xpg;

struct App {
    // USER: data
};

int main(int argc, char** argv) {
    app::Result result;
    app::Application application;
    result = app::CreateApplication(&application, {
        .minimum_api_version = VK_API_VERSION_1_1,
        .required_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING,
        .enable_validation_layer = true,
    });
    if (result != app::Result::SUCCESS) {
        log::error("Failed to initialize application\n");
        exit(100);
    }

    // USER: descriptors / shaders / pipelines / framegraphs

    // USER: application
    App app = {};

    application.key_event = [&app] () {};
    application.mouse_event = [&app] () {};
    application.resize = [&app] () {};
    application.gui = [&app] () {};
    application.render = [&app] () {};
    result = app::MainLoop(&application);

    if (result != app::Result::SUCCESS) {
        log::error("Main loop exited with error\n");
        exit(100);
    }

    // USER: cleanup
    app = {};

    app::DestroyApplication(&app);
}

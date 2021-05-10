// Customized Firefox configuration
pref("browser.startup.homepage", "https://courses.nvidia.com|http://localhost:8080/lab", locked);
pref("browser.startup.homepage_override.mstone", "ignore", locked);
pref("datareporting.policy.firstRunURL", "", locked);
pref("layers.acceleration.disabled", true, locked);
pref("media.hardware-video-decoding.enabled", false, locked);
// FF64 crashing with pipe error
pref("browser.tabs.remote.autostart", false, locked);

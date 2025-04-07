try {
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri https://sdk.lunarg.com/sdk/download/1.4.309.0/windows/VulkanSDK-1.4.309.0-Installer.exe -OutFile VulkanSDK.exe -UseBasicParsing
    ./VulkanSDK.exe --root C:/VulkanSDK --accept-licenses --default-answer --confirm-command install | Out-Null
    Remove-Item VulkanSDK.exe
}
catch
{
    Write-Output "Vulkan SDK installation failed";
    exit 1
}


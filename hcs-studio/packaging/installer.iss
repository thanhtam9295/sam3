; Inno Setup script: Windows installer for HCS Studio.
; Built by packaging/build_windows.ps1 after PyInstaller has made dist\HCS Studio.
;   iscc /DAppVersion=0.1.0 /DVariant=cpu packaging\installer.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Variant
  #define Variant "cpu"
#endif

[Setup]
AppId={{5B6F1A63-2C55-4E8B-9A57-3F0D1C2E7A41}
AppName=HCS Studio
AppVersion={#AppVersion}
AppPublisher=HCS Studio
DefaultDirName={autopf}\HCS Studio
DefaultGroupName=HCS Studio
DisableProgramGroupPage=yes
; Per-user install by default (no administrator rights needed); all users if elevated.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=HCS-Studio-{#AppVersion}-{#Variant}-setup
SetupIconFile=hcs_studio.ico
UninstallDisplayIcon={app}\HCS Studio.exe
Compression=lzma2/max
SolidCompression=yes
; Bundles with PyTorch (CUDA) exceed 2 GB: split into setup + .bin files.
DiskSpanning=yes
DiskSliceSize=max
WizardStyle=modern
ChangesEnvironment=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "addtopath"; Description: "Add the hcs-studio command to PATH"; GroupDescription: "Command line:"; Flags: unchecked

[Files]
Source: "..\dist\HCS Studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\HCS Studio"; Filename: "{app}\HCS Studio.exe"
Name: "{group}\HCS Studio user guide"; Filename: "{app}\docs\index.html"
Name: "{group}\Uninstall HCS Studio"; Filename: "{uninstallexe}"
Name: "{autodesktop}\HCS Studio"; Filename: "{app}\HCS Studio.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\HCS Studio.exe"; Description: "Start HCS Studio"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsAddPath(Dir: string): Boolean;
var
  Paths: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') = 0;
end;

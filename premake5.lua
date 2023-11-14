-- premake5.lua
workspace "RayTraceApp"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "RayTraceApp"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
include "Walnut/WalnutExternal.lua"

include "RayTraceApp"
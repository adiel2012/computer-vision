# Algorithm Index - Complete Reference

Quick reference for finding specific algorithms and techniques across all 20 chapters.

## 📐 Mathematical Foundations

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Vector Operations | 1 | Add, subtract, scale, dot product, cross product |
| Vector Normalization | 1 | Unit vector computation |
| Matrix Multiplication | 1 | 3x3 and 4x4 matrix operations |
| Matrix Inverse | 1 | Gauss-Jordan elimination |
| Matrix Transpose | 1 | Row-column swap |
| Quaternions | 1 | Rotation representation |
| SLERP | 1, 15 | Spherical linear interpolation |
| Gram-Schmidt | 1 | Orthonormalization |
| Homogeneous Coordinates | 1 | 4D representation for 3D transforms |

## 🎨 2D Graphics Algorithms

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Bresenham Line | 2 | Integer-only line drawing |
| DDA Line | 2 | Digital differential analyzer |
| Midpoint Circle | 2 | Circle rasterization |
| Bresenham Circle | 2 | Integer circle drawing |
| Scanline Fill | 2 | Polygon filling |
| Cohen-Sutherland | 2 | Line clipping |
| Liang-Barsky | 2 | Parametric line clipping |
| Sutherland-Hodgeman | 2 | Polygon clipping |

## 🔷 3D Geometry

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Bézier Curves | 3 | Parametric curve evaluation |
| De Casteljau | 3 | Stable Bézier computation |
| B-splines | 3 | Smooth curve interpolation |
| Catmull-Rom Splines | 3, 15 | Interpolating splines |
| Bézier Patches | 3 | Parametric surfaces |
| Signed Distance Functions | 3 | Implicit surfaces |
| Perlin Noise | 3, 13 | Procedural noise |
| Marching Cubes | 16 | Isosurface extraction |

## 🔄 Transformations

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Translation Matrix | 1, 4 | Position offset |
| Rotation Matrix (Axis) | 1, 4 | Rx, Ry, Rz rotation |
| Scaling Matrix | 1, 4 | Size transformation |
| Rodrigues' Formula | 1, 4 | Arbitrary axis rotation |
| Look-At Matrix | 4 | Camera view matrix |
| Perspective Projection | 4 | 3D to 2D projection |
| Orthographic Projection | 4 | Parallel projection |
| Viewport Transform | 4 | NDC to screen space |
| Normal Transformation | 4 | Inverse transpose for normals |

## 👁️ Visibility & Culling

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Z-Buffer | 5, 6 | Depth testing |
| Painter's Algorithm | 5 | Back-to-front sorting |
| Back-Face Culling | 5 | Eliminate back faces |
| Frustum Culling | 5 | View frustum testing |
| Occlusion Culling | 5 | Hidden object removal |
| BSP Tree | 5 | Binary space partitioning |
| Portal Culling | 5 | Room-based visibility |

## 🔺 Rasterization

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Barycentric Coordinates | 6 | Triangle point testing |
| Edge Function | 6 | Half-space test |
| Scanline Rasterization | 6 | Row-by-row filling |
| Tile-based Rasterization | 6 | Block-based rendering |
| Perspective-Correct Interpolation | 6 | Attribute interpolation |
| Top-Left Rule | 6 | Edge pixel ownership |
| Bounding Box | 6 | Triangle bounds |
| SSAA | 6, 17 | Supersampling anti-aliasing |

## 💡 Shading & Lighting

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Lambertian Reflection | 7, 12 | Diffuse shading |
| Phong Reflection | 7 | Specular highlights |
| Blinn-Phong | 7 | Modified specular |
| Flat Shading | 7 | Per-face shading |
| Gouraud Shading | 7 | Per-vertex interpolation |
| Phong Shading | 7 | Per-pixel lighting |
| Normal Mapping | 7, 13 | Surface detail |
| Bump Mapping | 13 | Height-based normals |
| Tangent Space | 7, 13 | Local coordinate system |

## 🌟 Ray Tracing - Core

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Ray-Sphere Intersection | 8 | Quadratic solution |
| Ray-Plane Intersection | 8 | Linear solution |
| Ray-Triangle (Möller-Trumbore) | 8 | Barycentric intersection |
| Ray-AABB | 8, 9 | Slab method |
| Ray-Disk | 8 | Circle intersection |
| Reflection Ray | 8, 10 | Mirror reflection |
| Refraction Ray | 8, 10 | Snell's law |
| Shadow Ray | 8 | Visibility testing |
| Recursive Ray Tracing | 8 | Depth-limited recursion |

## ⚡ Acceleration Structures

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| BVH Construction | 9 | Bounding volume hierarchy |
| SAH (Surface Area Heuristic) | 9 | Optimal BVH splits |
| BVH Traversal | 9 | Tree intersection |
| k-d Tree | 9 | Space partitioning |
| Octree | 9 | 8-way space division |
| Uniform Grid | 9 | Regular grid |
| 3D-DDA | 9 | Grid traversal |
| Mailboxing | 9 | Duplicate avoidance |

## 🎲 Monte Carlo & Sampling

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Random Sampling | 10 | Uniform random |
| Stratified Sampling | 10 | Jittered grid |
| Cosine-Weighted Sampling | 10, 11 | Hemisphere sampling |
| Random in Unit Sphere | 10 | 3D uniform distribution |
| Random in Unit Disk | 10 | 2D uniform distribution |
| Halton Sequence | 10, 19 | Low-discrepancy |
| Sobol Sequence | 19 | Quasi-random |
| Importance Sampling | 11, 12 | PDF-based sampling |
| Multiple Importance Sampling | 11 | Combining strategies |
| Russian Roulette | 11 | Path termination |

## 🌍 Global Illumination

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Path Tracing | 11 | Monte Carlo light transport |
| Direct Light Sampling | 11 | Explicit light connection |
| NEE (Next Event Estimation) | 11 | Light sampling |
| Bidirectional Path Tracing | 11, 17 | Eye + light paths |
| Photon Mapping | 14, 17 | Two-pass algorithm |
| Instant Radiosity | 17 | Virtual point lights |
| Metropolis Light Transport | 17 | MCMC sampling |
| Radiosity | 17 | Finite element method |

## 🎭 Physically Based Rendering

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Cook-Torrance BRDF | 12 | Microfacet model |
| GGX Distribution | 12 | Normal distribution |
| Beckmann Distribution | 12 | Alternative NDF |
| Smith Geometry | 12 | Shadowing-masking |
| Schlick Fresnel | 8, 12 | Fresnel approximation |
| Disney Principled BRDF | 12 | Artist-friendly PBR |
| Oren-Nayar | 12 | Rough diffuse |
| Ashikhmin-Shirley | 12 | Anisotropic specular |

## 🖼️ Texturing

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| UV Mapping | 13 | Texture coordinates |
| Nearest Neighbor | 13 | Point sampling |
| Bilinear Interpolation | 13 | 2x2 filter |
| Trilinear Interpolation | 13 | Mipmap interpolation |
| Mipmap Generation | 13 | LOD pyramid |
| Anisotropic Filtering | 13 | Directional blur |
| Planar Mapping | 13 | Flat projection |
| Cylindrical Mapping | 13 | Cylinder unwrap |
| Spherical Mapping | 13 | Sphere unwrap |
| Cubic Mapping | 13 | 6-face projection |
| Perlin Noise Texture | 13 | Procedural pattern |
| Worley Noise | 13 | Cellular texture |

## 🌫️ Advanced Rendering

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Ambient Occlusion | 14 | Hemisphere sampling |
| Screen-Space AO | 14 | Post-process AO |
| Subsurface Scattering | 14 | Diffusion approximation |
| Volumetric Rendering | 14 | Ray marching |
| Shadow Mapping | 14 | Depth-based shadows |
| PCF (Percentage Closer) | 14 | Soft shadow filtering |
| Bloom | 14, 17 | Bright pass + blur |
| Tone Mapping | 17 | HDR to LDR |
| Depth of Field | 10 | Lens simulation |
| Motion Blur | 10 | Temporal sampling |
| Caustics | 14, 17 | Photon mapping |

## 🏃 Animation & Physics

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Linear Interpolation (Lerp) | 15 | Position blending |
| Ease-In-Ease-Out | 15 | Smooth curves |
| Cubic Hermite | 15 | Smooth interpolation |
| Euler Integration | 15 | Simple physics |
| Verlet Integration | 15 | Stable physics |
| RK4 Integration | 15 | High-accuracy physics |
| Forward Kinematics | 15 | Joint hierarchy |
| Inverse Kinematics (CCD) | 15 | Target reaching |
| Linear Blend Skinning | 15 | Mesh deformation |
| Dual Quaternion Skinning | 15 | Better deformation |
| Collision Detection | 15 | Sphere/AABB tests |
| Collision Response | 15 | Impulse-based |
| Mass-Spring System | 15 | Cloth simulation |

## 🔨 Mesh Processing

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Edge Collapse | 16 | Simplification |
| Quadric Error Metric | 16 | QEM decimation |
| Catmull-Clark Subdivision | 16 | Smooth subdivision |
| Loop Subdivision | 16 | Triangle subdivision |
| Laplacian Smoothing | 16 | Mesh denoising |
| Taubin Smoothing | 16 | Non-shrinking smooth |
| Convex Hull (Gift Wrap) | 16 | 3D convex hull |
| Convex Hull (Quickhull) | 16 | Faster convex hull |
| Boolean Operations | 16 | CSG operations |
| Mesh Parameterization | 16 | UV unwrapping |

## ⚙️ Optimization

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Multi-threading | 18 | Parallel rendering |
| Thread Pool | 18 | Worker threads |
| Tile-based Parallelism | 18 | Image tiles |
| SIMD Vectorization | 18 | SSE/AVX |
| Ray Packet Tracing | 18 | Coherent rays |
| Cache-Friendly Layout | 18 | SoA vs AoS |
| CUDA Kernels | 18 | GPU parallelism |
| Warp Efficiency | 18 | GPU optimization |
| Shared Memory | 18 | GPU cache |
| Atomic Operations | 18 | Thread-safe ops |

## 🔬 Advanced Ray Tracing

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Real-time Ray Tracing | 19 | Low sample RT |
| Temporal Accumulation | 19 | Frame blending |
| Spatial Denoising | 19 | NLM filter |
| Wavelet Denoising | 19 | Multi-scale filter |
| SVGF Denoising | 19 | Spatiotemporal |
| Reservoir Sampling | 19 | ReSTIR |
| Hybrid Rendering | 19 | Raster + RT |
| Adaptive Sampling | 19 | Variance-based |
| Blue Noise Sampling | 19 | Error diffusion |

## 📊 Image Processing

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Gaussian Blur | 14, 17 | Separable filter |
| Box Blur | 14 | Fast approximation |
| Bilateral Filter | 14 | Edge-preserving |
| Median Filter | 17 | Noise reduction |
| Sobel Edge Detection | 17 | Gradient operator |
| Reinhard Tone Mapping | 17 | Simple HDR |
| ACES Tone Mapping | 17 | Filmic curve |
| Gamma Correction | 7, 17 | sRGB conversion |
| Color Grading | 17 | LUT application |

## 🎯 Scene Management

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| Scene Graph | 20 | Hierarchical organization |
| Frustum Culling | 5, 20 | View frustum test |
| LOD Selection | 16, 20 | Level of detail |
| Spatial Hashing | 9, 20 | Object queries |
| Render Queue | 20 | Draw call sorting |
| Material Batching | 20 | Reduce state changes |

## 📚 File I/O & Parsing

| Algorithm | Chapter | Description |
|-----------|---------|-------------|
| PPM Writer | 2, 20 | Simple image format |
| BMP Writer | 20 | Windows bitmap |
| PNG Writer | 20 | Compressed images |
| OBJ Parser | 3, 20 | Mesh loading |
| JSON Parser | 20 | Scene description |

## 🔍 Quick Lookup by Use Case

### Need to render triangles?
→ Ch 6: Barycentric coordinates, edge functions

### Need realistic lighting?
→ Ch 11-12: Path tracing, Cook-Torrance BRDF

### Need it to run fast?
→ Ch 9: BVH acceleration
→ Ch 18: Multi-threading, SIMD, CUDA

### Need soft shadows?
→ Ch 10: Distribution ray tracing with area lights

### Need reflections?
→ Ch 8: Recursive ray tracing
→ Ch 12: Fresnel equations

### Need textures?
→ Ch 13: UV mapping, bilinear filtering, mipmaps

### Need motion?
→ Ch 15: Keyframe animation, physics simulation

### Need to edit meshes?
→ Ch 16: Subdivision, simplification, smoothing

### Need production quality?
→ Ch 14: AO, SSS, volumetrics
→ Ch 17: Advanced techniques
→ Ch 19: Denoising

## 📖 Algorithm Complexity Reference

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Bresenham Line | O(n) | O(1) |
| BVH Construction (SAH) | O(n log²n) | O(n) |
| BVH Traversal | O(log n) | O(log n) stack |
| Ray-Triangle | O(1) | O(1) |
| Barycentric | O(1) | O(1) |
| Matrix Multiply | O(n³) general, O(1) for 4x4 | O(1) for fixed |
| Gaussian Blur | O(wh·k²) | O(wh) |
| Path Tracing | O(spp · depth · n) | O(depth) stack |
| Catmull-Clark | O(n) per iteration | O(n) |
| Marching Cubes | O(n³) | O(n²) |

Where:
- n = number of objects/triangles
- w, h = image width, height
- k = kernel size
- spp = samples per pixel
- depth = ray bounce depth

---

**Use Ctrl+F to search for specific algorithms!**

This index covers all major algorithms across the 20 chapters.

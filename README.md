# 3D Computer Graphics - Comprehensive Tutorial
## From-Scratch Implementation Guide

## Overview
This comprehensive tutorial focuses on classical 3D computer graphics with an emphasis on building everything from first principles without relying on external graphics libraries. You will implement all rendering algorithms, data structures, and mathematical operations from scratch. Based on principal textbooks including:
- **Computer Graphics: Principles and Practice (3rd Ed.)** by John F. Hughes, Andries van Dam, Morgan McGuire, et al.
- **Fundamentals of Computer Graphics (5th Ed.)** by Steve Marschner and Peter Shirley
- **Physically Based Rendering: From Theory to Implementation (4th Ed.)** by Matt Pharr, Wenzel Jakob, and Greg Humphreys
- **Computer Graphics from Scratch** by Gabriel Gambetta

**Tutorial Philosophy:** No OpenGL, DirectX, or graphics APIs. All rendering code written from first principles to deeply understand how graphics systems work internally. CUDA can be optionally used for performance optimization.

---

## Chapter Structure

Each chapter has a corresponding Jupyter notebook for implementation. You can open them directly in Google Colab:

| Chapter | Notebook | Open in Colab |
|---------|----------|---------------|
| 1 | Mathematical Foundations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_01_math_foundations.ipynb) |
| 2 | Framebuffer and 2D Rasterization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_02_framebuffer_2d.ipynb) |
| 3 | 3D Geometry and Data Structures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_03_3d_geometry.ipynb) |
| 4 | 3D Transformations and Viewing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_04_transformations.ipynb) |
| 5 | Visibility and Hidden Surface Removal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_05_visibility_and_hidden_surface_removal.ipynb) |
| 6 | Triangle Rasterization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_06_triangle_rasterization.ipynb) |
| 7 | Shading and Illumination | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_07_shading_and_illumination.ipynb) |
| 8 | Ray Tracing - Core | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_08_ray_tracing_-_core_implementation.ipynb) |
| 9 | Acceleration Structures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_09_acceleration_structures.ipynb) |
| 10 | Distribution Ray Tracing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_10_distribution_ray_tracing_and_monte_carlo.ipynb) |
| 11 | Path Tracing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_11_path_tracing_implementation.ipynb) |
| 12 | Physically Based Rendering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_12_physically_based_rendering_pbr.ipynb) |
| 13 | Texturing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_13_texturing_from_scratch.ipynb) |
| 14 | Advanced Rendering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_14_advanced_rendering_techniques.ipynb) |
| 15 | Animation and Simulation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_15_animation_and_simulation.ipynb) |
| 16 | Geometric Algorithms | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_16_geometric_algorithms.ipynb) |
| 17 | Advanced Topics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_17_advanced_topics.ipynb) |
| 18 | Optimization and Parallelization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_18_optimization_and_parallelization.ipynb) |
| 19 | Advanced Ray Tracing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_19_advanced_ray_tracing_topics.ipynb) |
| 20 | Integration and Projects | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_20_integration_and_projects.ipynb) |

> **Note:** After pushing to GitHub, the Colab badges above will automatically open the notebooks in Google Colab for interactive learning.

---

## Chapter 1: Mathematical Foundations and Implementation

### Week 1-3: Building Math Libraries from Scratch
- **Vector Implementation**
  - Implementing Vec2, Vec3, Vec4 classes
  - Vector operations (addition, subtraction, scalar multiplication)
  - Dot product and cross product implementation
  - Vector length and normalization algorithms
  - Orthogonal and orthonormal bases
  - Gram-Schmidt orthogonalization

- **Matrix Implementation**
  - Matrix3x3 and Matrix4x4 classes
  - Matrix multiplication (naive and optimized)
  - Transpose and inverse algorithms
  - Determinant calculation
  - LU decomposition
  - Solving linear systems

- **Coordinate Systems**
  - Homogeneous coordinates representation
  - Object space, world space, camera space, screen space
  - Change of basis implementation
  - Coordinate system handedness

- **Transformation Matrices**
  - Translation matrix construction
  - Rotation matrices (around x, y, z axes)
  - Scaling matrices
  - General rotation (arbitrary axis via Rodrigues' formula)
  - Transformation composition
  - Inverse transformations

- **Rotation Representations**
  - Euler angles implementation
  - Axis-angle representation
  - Quaternion class (w, x, y, z)
  - Quaternion operations (multiply, conjugate, inverse)
  - SLERP (Spherical Linear Interpolation)
  - Conversion between representations

**Key References:** Marschner & Shirley Ch. 2, 5-6, Gambetta Ch. 1

---

## Chapter 2: Framebuffer and 2D Rasterization

### Week 4-5: Building the Foundation
- **Framebuffer Implementation**
  - 2D array for pixel storage
  - Color representation (RGB, RGBA)
  - Pixel addressing and access
  - Writing to image files (PPM, BMP format)
  - Double buffering concept
  - Clearing the framebuffer

- **Basic 2D Drawing Primitives**
  - Pixel plotting
  - Line drawing algorithms:
    - DDA (Digital Differential Analyzer) implementation
    - Bresenham's line algorithm
    - Midpoint line algorithm
  - Circle and ellipse drawing:
    - Midpoint circle algorithm
    - Bresenham's circle algorithm
  - Anti-aliased lines (basic)

- **Polygon Filling**
  - Scan-line polygon fill algorithm
  - Edge table and active edge table construction
  - Filling arbitrary polygons
  - Triangle filling (special case)

- **2D Clipping**
  - Cohen-Sutherland line clipping
  - Liang-Barsky line clipping
  - Sutherland-Hodgeman polygon clipping
  - Implementation details and edge cases

**Key References:** Marschner & Shirley Ch. 3, 8, Gambetta Ch. 2-3

---

## Chapter 3: 3D Geometry and Data Structures

### Week 6-8: Geometric Representations
- **3D Primitive Structures**
  - Point3D class
  - Ray class (origin + direction)
  - Plane representation
  - Triangle structure (vertices, normals, UVs)
  - Sphere, cylinder, cone definitions
  - Bounding volumes (AABB, OBB, Sphere)

- **Mesh Data Structures**
  - Vertex array
  - Triangle/face array
  - Indexed mesh representation
  - Normal storage (per-vertex vs. per-face)
  - UV coordinate storage
  - Simple mesh file loading (OBJ format parser)

- **Parametric Curves (Implementation)**
  - Bézier curve evaluation
  - De Casteljau's algorithm
  - B-spline curve evaluation
  - Curve tessellation to line segments
  - Catmull-Rom spline implementation

- **Parametric Surfaces**
  - Bézier patch evaluation
  - B-spline surface evaluation
  - Surface tessellation to triangle meshes
  - Normal computation for parametric surfaces

- **Implicit Surfaces**
  - Signed distance function (SDF) implementation
  - Sphere SDF
  - Box SDF
  - Metaball evaluation
  - Ray marching for implicit surfaces

- **Procedural Geometry**
  - Procedural mesh generation (sphere, cube, cylinder)
  - Perlin noise implementation from scratch
  - Terrain generation using noise
  - Fractal algorithms (recursive subdivision)

**Key References:** Marschner & Shirley Ch. 12, 15, Gambetta Ch. 4

---

## Chapter 4: 3D Transformations and Viewing Pipeline

### Week 9-11: The Complete Pipeline Implementation
- **3D Transformation Implementation**
  - Translation matrix construction
  - Rotation matrices (Rx, Ry, Rz)
  - Scaling matrix
  - Arbitrary axis rotation (Rodrigues)
  - Affine transformation class
  - Transformation composition and optimization
  - Normal transformation (inverse transpose)

- **Camera System Implementation**
  - Camera class (position, target, up vector)
  - Look-at matrix construction from scratch
  - View matrix derivation and implementation
  - First-person and orbit camera controls
  - Camera coordinate system

- **Projection Implementation**
  - Orthographic projection matrix
  - Perspective projection matrix derivation
  - Field of view (FOV) calculations
  - Perspective divide implementation
  - Depth calculation and storage

- **Viewport Transformation**
  - NDC (Normalized Device Coordinates) to screen space
  - Viewport matrix implementation
  - Aspect ratio handling
  - Window coordinate transformation

- **Complete Transformation Pipeline**
  - Model → World → View → Clip → NDC → Screen
  - Pipeline class implementation
  - Vertex transformation function
  - Homogeneous clipping against frustum planes
  - Perspective-correct attribute interpolation

**Key References:** Marschner & Shirley Ch. 7-8, Gambetta Ch. 5-6

---

## Chapter 5: Visibility and Hidden Surface Removal

### Week 12-13: Culling and Depth
- **Back-Face Culling Implementation**
  - Normal computation
  - View-dependent culling test
  - Winding order determination
  - Early rejection optimization

- **Frustum Culling**
  - Frustum plane extraction from projection matrix
  - Point-plane distance test
  - Sphere-frustum test
  - AABB-frustum test
  - Hierarchical culling with bounding volumes

- **Z-Buffer Algorithm**
  - Depth buffer implementation (2D array)
  - Per-pixel depth testing
  - Depth buffer initialization
  - Z-fighting and precision issues
  - W-buffering alternative

- **Painter's Algorithm**
  - Depth sorting
  - Back-to-front rendering
  - Handling intersecting polygons
  - Limitations

- **BSP Tree Implementation**
  - Binary Space Partitioning tree construction
  - Polygon splitting algorithm
  - BSP tree traversal for rendering
  - Front-to-back vs. back-to-front

**Key References:** Marschner & Shirley Ch. 8, Gambetta Ch. 7

---

## Chapter 6: Triangle Rasterization

### Week 14-16: Software Rasterizer
- **Triangle Rasterization Algorithms**
  - Edge function method
  - Barycentric coordinates calculation
  - Scanline rasterization
  - Tile-based rasterization approach
  - Rasterization rules (top-left rule)

- **Attribute Interpolation**
  - Linear interpolation implementation
  - Perspective-correct interpolation
  - Interpolating colors, normals, UVs
  - Depth interpolation
  - Barycentric interpolation formula

- **Optimizations**
  - Bounding box calculation
  - Early rejection tests
  - Incremental edge updates
  - Fixed-point arithmetic
  - SIMD considerations (optional)

- **Anti-Aliasing**
  - Supersampling (SSAA) implementation
  - Jittered sampling
  - Coverage calculation
  - Sample accumulation and averaging

**Key References:** Marschner & Shirley Ch. 8, Gambetta Ch. 8

---

## Chapter 7: Shading and Illumination

### Week 17-19: Lighting Models
- **Color Implementation**
  - RGB color class
  - Color operations (add, multiply, clamp)
  - Gamma correction functions
  - Linear vs. sRGB color space
  - Color conversion utilities

- **Light Source Classes**
  - Point light implementation
  - Directional light
  - Spotlight (cone calculations)
  - Area light representation
  - Ambient light
  - Light attenuation calculation

- **Reflection Models Implementation**
  - Lambertian diffuse reflection
  - Phong specular reflection
  - Blinn-Phong model
  - Combined illumination calculation
  - Material property structure

- **Shading Algorithms**
  - Flat shading implementation
  - Gouraud shading (per-vertex)
  - Phong shading (per-pixel)
  - Normal interpolation
  - Lighting calculation per fragment

- **Normal Mapping**
  - Tangent space calculation
  - TBN matrix construction
  - Normal map sampling and application
  - Perturbed normal calculation

**Key References:** Marschner & Shirley Ch. 18-20, Gambetta Ch. 9-10

---

## Chapter 8: Ray Tracing - Core Implementation

### Week 20-24: Building a Ray Tracer from Scratch
- **Ray Tracer Architecture**
  - Ray class (origin, direction, t_min, t_max)
  - Scene representation
  - Camera ray generation
  - Main rendering loop
  - Pixel sampling strategy

- **Ray-Object Intersection Implementation**
  - Ray-sphere intersection
    - Quadratic equation solution
    - Discriminant calculation
    - Near/far hit determination
  - Ray-plane intersection
  - Ray-triangle intersection
    - Möller-Trumbore algorithm implementation
    - Barycentric coordinate calculation
  - Ray-box intersection (slab method)
  - Ray-disk intersection
  - Normal computation at hit point

- **Recursive Ray Tracing**
  - Reflection ray generation
  - Refraction ray generation
    - Snell's law implementation
    - Total internal reflection check
    - Fresnel equations (Schlick approximation)
  - Recursive trace function
  - Depth limiting
  - Ray color accumulation

- **Shadow Rays**
  - Shadow ray generation
  - Shadow testing against all lights
  - Hard shadow implementation
  - Epsilon offset to prevent self-intersection
  - Shadow acne fixes

- **Local Illumination in Ray Tracer**
  - Phong/Blinn-Phong at hit point
  - Multiple light source support
  - Ambient, diffuse, specular components
  - Material properties

**Key References:** Marschner & Shirley Ch. 4, 13, Pharr et al. Ch. 1-2, 6-7, Gambetta Ch. 11-13, Shirley "Ray Tracing in One Weekend"

---

## Chapter 9: Acceleration Structures

### Week 25-27: Optimizing Ray Tracing
- **Bounding Volume Hierarchy (BVH)**
  - AABB implementation (min, max points)
  - BVH node structure (internal vs. leaf)
  - BVH construction algorithms:
    - Top-down recursive construction
    - Object partitioning strategies
    - Midpoint split
    - Surface Area Heuristic (SAH)
    - Binned SAH
  - BVH traversal algorithm
  - Stack-based vs. recursive traversal
  - Optimizing memory layout

- **Spatial Subdivision**
  - Uniform grid implementation:
    - Grid cell calculation
    - Object assignment to cells
    - Ray traversal (3D-DDA)
  - Octree implementation:
    - Recursive subdivision
    - Object insertion
    - Ray traversal
  - k-d tree:
    - Axis-aligned splitting planes
    - Construction algorithms
    - Traversal with near/far ordering

- **Optimization Techniques**
  - Early ray termination
  - Mailboxing to avoid duplicate tests
  - Ray packet tracing
  - Bounding volume tightening

**Key References:** Pharr et al. Ch. 4, Marschner & Shirley Ch. 13

---

## Chapter 10: Distribution Ray Tracing and Monte Carlo

### Week 28-31: Advanced Ray Tracing
- **Random Number Generation**
  - Implementing a good RNG (LCG, PCG, Mersenne Twister)
  - Uniform random numbers [0, 1)
  - Random point in unit sphere
  - Random point in unit disk
  - Cosine-weighted hemisphere sampling

- **Distribution Ray Tracing Implementation**
  - Soft shadows from area lights:
    - Random sampling on light surface
    - Multiple shadow rays
    - Shadow ray accumulation
  - Glossy reflections:
    - Perturbed reflection direction
    - Microfacet roughness
  - Translucency (rough refractions)
  - Depth of field:
    - Lens aperture simulation
    - Focal plane
    - Random sampling on lens
  - Motion blur:
    - Time sampling
    - Object transformation interpolation

- **Monte Carlo Integration Basics**
  - Monte Carlo estimator implementation
  - Computing π via Monte Carlo (example)
  - Variance and standard error
  - Importance sampling concept

- **Sampling Techniques**
  - Uniform sampling
  - Stratified sampling implementation
  - Jittered sampling
  - Multi-jittered sampling
  - Low-discrepancy sequences (Halton sequence)

**Key References:** Pharr et al. Ch. 7, 13, Marschner & Shirley Ch. 14, Shirley "Ray Tracing: The Rest of Your Life"

---

## Chapter 11: Path Tracing Implementation

### Week 32-35: Global Illumination from Scratch
- **The Rendering Equation**
  - Understanding the rendering equation
  - Direct vs. indirect illumination
  - Exitant radiance calculation
  - Recursive formulation

- **Path Tracing Algorithm**
  - Recursive path tracing implementation
  - Random ray direction generation (cosine-weighted)
  - Russian roulette for termination
  - Path throughput accumulation
  - Radiance estimation

- **Direct Light Sampling**
  - Explicit light sampling
  - Shadow ray to random light point
  - Combining direct and indirect illumination

- **Multiple Importance Sampling (MIS)**
  - MIS concept and motivation
  - Power heuristic implementation
  - Combining BRDF sampling and light sampling
  - Balance heuristic

- **Bidirectional Path Tracing (Conceptual)**
  - Light paths and eye paths
  - Path connection strategy
  - Implementation complexity
  - When to use BDPT

- **Progressive Rendering**
  - Sample accumulation
  - Progressive refinement
  - Real-time preview updates
  - Convergence monitoring

**Key References:** Pharr et al. Ch. 13-16, Marschner & Shirley Ch. 24, Shirley series

---

## Chapter 12: Physically Based Rendering (PBR)

### Week 36-39: BRDF and Material System
- **BRDF Implementation**
  - BRDF interface/abstract class
  - Lambertian BRDF implementation
  - Perfect specular (mirror) BRDF
  - Blinn-Phong BRDF

- **Microfacet Theory**
  - Microfacet BRDF structure
  - Normal Distribution Function (NDF):
    - Beckmann distribution
    - GGX (Trowbridge-Reitz) implementation
  - Geometry/Shadowing term:
    - Smith G1 implementation
    - Height-correlated Smith G2
  - Fresnel term:
    - Exact Fresnel equations
    - Schlick approximation implementation

- **Cook-Torrance BRDF**
  - Full Cook-Torrance implementation
  - Metallic vs. dielectric materials
  - Energy conservation
  - Importance sampling the BRDF
  - PDF calculation

- **Material System**
  - Material class hierarchy
  - Diffuse material
  - Metal material
  - Dielectric (glass) material
  - Plastic (layered) material
  - Material parameters (albedo, roughness, metallic, IOR)

- **Disney Principled BRDF**
  - Disney BRDF parameters
  - Implementation from the SIGGRAPH paper
  - Artist-friendly parameterization

- **Subsurface Scattering (Basic)**
  - Diffusion approximation
  - Dipole model (conceptual)
  - Simple SSS implementation

**Key References:** Pharr et al. Ch. 8-9, 14, Marschner & Shirley Ch. 14

---

## Chapter 13: Texturing from Scratch

### Week 40-42: Texture System Implementation
- **Texture Class Implementation**
  - 2D texture array (width × height × channels)
  - Texture loading from file (PPM, simple BMP)
  - Texel access and addressing
  - Texture coordinate wrapping (repeat, clamp, mirror)

- **Texture Sampling**
  - Nearest neighbor sampling
  - Bilinear interpolation implementation
  - UV coordinate clamping/wrapping
  - Texture coordinate transformation

- **Mipmapping**
  - Mipmap generation (box filter)
  - Level selection based on derivatives
  - Trilinear interpolation
  - Anisotropic filtering (basic)
  - Summed-area tables

- **UV Coordinate Generation**
  - Planar mapping
  - Cylindrical mapping
  - Spherical mapping (latitude-longitude)
  - Cubic mapping

- **Procedural Textures**
  - Checkerboard pattern
  - Perlin noise textures
  - Turbulence (summed octaves)
  - Marble, wood patterns
  - Worley/cellular noise

- **Normal and Bump Mapping**
  - Tangent space calculation
  - TBN matrix construction
  - Normal map interpretation
  - Bump mapping via finite differences

- **Environment Mapping**
  - Spherical environment maps
  - Cube maps implementation
  - Reflection/refraction mapping
  - Environment map sampling

**Key References:** Marschner & Shirley Ch. 11, Pharr et al. Ch. 10

---

## Chapter 14: Advanced Rendering Techniques

### Week 43-45: Shadows, AO, and More
- **Shadow Algorithms for Rasterization**
  - Shadow mapping:
    - Rendering depth from light's POV
    - Shadow map storage
    - Shadow testing during rendering
    - Bias and self-shadowing issues
  - Percentage Closer Filtering (PCF) implementation
  - Soft shadows via PCF

- **Ambient Occlusion**
  - Ray-traced ambient occlusion
  - Hemisphere sampling
  - Occlusion ray testing
  - AO factor calculation
  - Precomputed AO (baking)

- **Caustics (Ray Tracing)**
  - Photon mapping basics:
    - Photon structure (position, direction, power)
    - Photon tracing from lights
    - Photon storage (k-d tree)
    - Radiance estimation
    - Final gathering
  - Caustic photon map
  - Visualization

- **Volume Rendering**
  - Participating media implementation
  - Ray marching through volume
  - Absorption and scattering calculations
  - Transmittance computation
  - Simple volumetric path tracing

**Key References:** Pharr et al. Ch. 11, 15, 16, Marschner & Shirley Ch. 11, 14

---

## Chapter 15: Animation and Simulation

### Week 46-48: Motion and Dynamics
- **Keyframe Animation**
  - Keyframe data structure
  - Linear interpolation (lerp)
  - Cubic interpolation (Catmull-Rom, Hermite)
  - Quaternion SLERP for rotations
  - Animation timeline and playback

- **Skeletal Animation**
  - Bone/joint hierarchy representation
  - Forward kinematics implementation
  - Pose calculation
  - Skinning matrix computation
  - Linear blend skinning (LBS)
  - Dual quaternion skinning implementation

- **Inverse Kinematics (IK)**
  - 2-bone IK (analytic solution)
  - CCD (Cyclic Coordinate Descent) implementation
  - FABRIK algorithm
  - Joint constraints

- **Particle Systems**
  - Particle structure (position, velocity, life)
  - Emitter implementation
  - Force accumulation (gravity, wind)
  - Numerical integration (Euler, Verlet)
  - Particle rendering

- **Rigid Body Simulation (Basic)**
  - Physics state (position, orientation, velocity, angular velocity)
  - Force and torque accumulation
  - Euler integration
  - Collision detection (sphere-sphere, AABB)
  - Collision response (impulse-based)

- **Cloth Simulation (Simple)**
  - Mass-spring system
  - Particle constraints
  - Verlet integration
  - Collision handling

**Key References:** Marschner & Shirley Ch. 16-17

---

## Chapter 16: Geometric Algorithms

### Week 49-51: Mesh Processing
- **Mesh Simplification**
  - Edge collapse algorithm
  - Quadric error metric (QEM) implementation
  - Priority queue for edge selection
  - LOD generation

- **Mesh Subdivision**
  - Catmull-Clark subdivision implementation:
    - Face points, edge points, vertex points
    - New mesh topology construction
  - Loop subdivision for triangles
  - Adaptive subdivision

- **Mesh Smoothing**
  - Laplacian smoothing implementation
  - Computing Laplacian operator
  - Taubin smoothing (low-pass filter)

- **Mesh Parameterization (Basic)**
  - Planar UV unwrapping
  - Least-squares conformal mapping (conceptual)
  - Texture coordinate generation

- **Boolean Operations**
  - CSG tree representation
  - Ray tracing CSG primitives
  - Mesh-based booleans (conceptual)

- **Surface Reconstruction**
  - Point cloud to mesh (conceptual)
  - Marching cubes implementation (for implicit surfaces)
  - Isosurface extraction

**Key References:** Marschner & Shirley Ch. 12, Botsch et al. Polygon Mesh Processing

---

## Chapter 17: Advanced Topics

### Week 52-54: Specialized Techniques
- **Deferred Shading (Software)**
  - G-buffer implementation (multiple framebuffers)
  - Geometry pass (write to G-buffer)
  - Lighting pass (read from G-buffer)
  - Light accumulation
  - Benefits and limitations

- **Image-Based Post-Processing**
  - Bloom:
    - Bright pass extraction
    - Gaussian blur implementation (separable)
    - Additive blending
  - Tone mapping operators:
    - Reinhard tone mapping
    - Filmic curve
  - Color grading (simple LUT)

- **Anti-Aliasing Techniques**
  - Supersampling (SSAA)
  - Jittered sampling
  - Edge detection for selective AA
  - Temporal accumulation

- **Non-Photorealistic Rendering (NPR)**
  - Silhouette edge detection
  - Toon/cel shading
  - Hatching patterns
  - Outline rendering

- **Atmospheric Effects**
  - Simple fog (depth-based)
  - Atmospheric scattering model (Rayleigh)
  - Sky color calculation
  - Aerial perspective

**Key References:** Marschner & Shirley Ch. 14, Pharr et al. Ch. 8

---

## Chapter 18: Optimization and Parallelization

### Week 55-57: Performance
- **Profiling and Measurement**
  - Timing code sections
  - Identifying bottlenecks
  - Memory usage analysis
  - Cache-friendly data structures

- **Algorithmic Optimizations**
  - Bounding volume hierarchies revisited
  - Spatial data structure tuning
  - Early rejection strategies
  - Incremental computation

- **Data Structure Optimization**
  - Structure of Arrays (SoA) vs. Array of Structures (AoS)
  - Memory layout for cache efficiency
  - Alignment and padding

- **Multi-Threading**
  - Thread pool implementation
  - Tile-based parallelization
  - Scanline parallelization
  - Thread synchronization (mutexes, atomics)
  - Lock-free data structures (basics)

- **SIMD Basics**
  - Vector operations with SIMD intrinsics (optional)
  - Ray-AABB intersection with SIMD
  - Ray packet tracing

- **GPU Acceleration with CUDA (Optional)**
  - CUDA kernel basics
  - Parallel ray tracing on GPU
  - BVH traversal on GPU
  - Texture sampling on GPU
  - Memory management (global, shared, constant)
  - Optimization techniques for CUDA

- **Fixed-Point Arithmetic**
  - Fixed-point representation
  - Fixed-point operations for rasterization
  - Integer-only implementations

**Key References:** Pharr et al. Ch. 1 (performance), Optimization texts

---

## Chapter 19: Advanced Ray Tracing Topics

### Week 58-60: Cutting-Edge Techniques
- **Metropolis Light Transport (Conceptual)**
  - Markov Chain Monte Carlo
  - Mutation strategies
  - Implementation challenges

- **Bidirectional Path Tracing (BDPT)**
  - Eye path construction
  - Light path construction
  - Path connection
  - MIS weights for all paths

- **Photon Mapping Advanced**
  - Global photon map
  - Caustic photon map
  - Final gathering implementation
  - Progressive photon mapping

- **Instant Radiosity**
  - Virtual point light generation
  - Direct illumination of VPLs
  - Clamping for stability

- **Advanced Sampling**
  - Sobol sequences
  - Halton sequences
  - Stratified sampling in multiple dimensions
  - Quasi-Monte Carlo methods

**Key References:** Pharr et al. Ch. 16, Advanced rendering papers

---

## Chapter 20: Integration and Projects

### Week 61-65: Building Complete Systems
- **Scene Description and Loading**
  - Scene file format design (custom or simple JSON)
  - Scene parser implementation
  - Mesh loading (OBJ, PLY)
  - Texture loading
  - Material definitions
  - Light placement
  - Camera parameters

- **Complete Rendering Systems**
  - Hybrid renderer (rasterization + ray tracing)
  - Choosing between techniques per-object
  - Multi-pass rendering
  - Render pipeline architecture

- **Output and File Formats**
  - Writing PPM (text and binary)
  - Writing BMP
  - Writing PNG (using minimal library or from scratch)
  - HDR image formats (Radiance .hdr)

- **Interactive Rendering**
  - Simple window management (platform-specific or minimal SDL for display only)
  - Mouse/keyboard input
  - Camera controls
  - Real-time progressive rendering
  - Interactive path tracer

- **Debugging and Visualization**
  - Debug rendering modes (normals, depth, UVs, wireframe)
  - Bounding volume visualization
  - Ray visualization
  - Heatmaps for performance

**Key References:** Pharr et al. Ch. 1, System design principles

---

## Prerequisites

### Required Background
- **Strong Programming**: C++ or C proficiency (modern C++ preferred)
- **Linear Algebra**: Vectors, matrices, transformations
- **Calculus**: Derivatives, integrals, multivariable calculus
- **Data Structures**: Arrays, trees, spatial structures
- **Algorithms**: Sorting, searching, recursion, complexity analysis

### Recommended Background
- Numerical methods and algorithms
- Physics (optics, mechanics)
- Computer architecture (caching, parallelism)

---

## Recommended Textbooks

### Primary Texts

1. **Marschner, Steve & Shirley, Peter.** *Fundamentals of Computer Graphics* (5th Edition), 2021
   - Excellent for algorithms and theory
   - Implementation-focused approach

2. **Pharr, Matt, Jakob, Wenzel & Humphreys, Greg.** *Physically Based Rendering: From Theory to Implementation* (4th Edition), 2023
   - Complete ray tracer implementation
   - Best practices for production code
   - Available online for free

3. **Gambetta, Gabriel.** *Computer Graphics from Scratch*, 2021
   - No dependencies approach
   - Step-by-step implementation
   - Perfect for this course philosophy

4. **Shirley, Peter.** *Ray Tracing in One Weekend* series
   - Hands-on ray tracing
   - Minimal dependencies
   - Great starting point

### Supplementary Texts

5. **Hughes, John F., van Dam, Andries, McGuire, Morgan, et al.** *Computer Graphics: Principles and Practice* (3rd Edition), 2013
   - Comprehensive reference
   - Algorithm details

6. **Glassner, Andrew S.** *An Introduction to Ray Tracing*, 1989
   - Classic ray tracing text
   - Fundamental concepts

### Online Resources

7. **Scratchapixel** (www.scratchapixel.com)
   - Free tutorials
   - Implementation-focused
   - No graphics API dependencies

8. **Ray Tracing in One Weekend** (raytracing.github.io)
   - Free online book series
   - Complete implementations

9. **PBR Book** (pbr-book.org)
   - Free online version of Pharr et al.
   - Complete source code

---

## Software and Tools

### Minimal Required Software

**Core Development:**
- **C++ compiler**: GCC, Clang, or MSVC (C++17 or later)
- **Build system**: CMake or Make
- **Text editor/IDE**: Visual Studio, VS Code, CLion, Vim, Emacs

**Version Control:**
- **Git** for source control

**Image Viewing:**
- Any image viewer that supports PPM, BMP, PNG

### Optional Utilities (Minimal)

**For Window Display Only (not for rendering):**
- **SDL2** - minimal usage, only for displaying framebuffer pixels
- **GLFW** - only for window creation and input
- **Platform-specific APIs** - Win32, X11, Cocoa (just for window)

**For Image I/O (if not implementing from scratch):**
- **stb_image.h** and **stb_image_write.h** - single-header libraries
- **Custom PPM/BMP implementation** (preferred)

**GPU Programming (Optional):**
- **CUDA Toolkit** - for GPU acceleration
- **NVIDIA GPU** - for CUDA support

**Debugging:**
- **GDB** or **LLDB** - debuggers
- **Valgrind** - memory analysis
- **Nsight Compute** - CUDA profiling (if using CUDA)
- **Custom debug visualization tools**

### Explicitly NOT Used

- ❌ OpenGL / GLSL
- ❌ DirectX / HLSL
- ❌ Vulkan / SPIR-V
- ❌ Metal
- ❌ WebGL
- ❌ Any GPU APIs
- ❌ Graphics shader libraries
- ❌ Hardware-accelerated rendering

---

## What You Will Learn

By completing this tutorial, you will be able to:

1. **Implement fundamental graphics algorithms** from scratch:
   - Rasterization pipeline (line drawing, triangle filling, clipping)
   - Z-buffer visibility
   - Transformation pipeline
   - Ray tracing (intersection, reflection, refraction)
   - Path tracing and global illumination

2. **Build complete math libraries** for graphics:
   - Vector and matrix operations
   - Transformation matrices
   - Quaternions for rotation
   - All mathematical operations needed for rendering

3. **Understand rendering deeply**:
   - How pixels are generated from 3D geometry
   - How light transport works physically
   - Trade-offs between different rendering approaches
   - Where performance bottlenecks occur

4. **Implement data structures** for graphics:
   - BVH, k-d trees, octrees
   - Mesh representations
   - Scene graphs
   - Spatial subdivision

5. **Optimize rendering code**:
   - Algorithmic optimizations
   - Cache-friendly data structures
   - Multi-threading
   - Profiling and measurement

6. **Build production-quality renderers**:
   - Software rasterizer
   - Ray tracer
   - Path tracer
   - Hybrid systems

7. **Read and implement** research papers in rendering

8. **Appreciate graphics APIs** by understanding what they do under the hood

---

## Tutorial Structure Overview

### Part I: Foundations (Modules 1-7)
- Math library implementation
- Framebuffer and 2D rasterization
- 3D geometry and transformations
- Complete software rasterizer
- Shading and illumination

### Part II: Ray Tracing (Modules 8-12)
- Basic ray tracer
- Acceleration structures
- Distribution ray tracing
- Monte Carlo integration
- Path tracing
- Physically based rendering

### Part III: Advanced Topics (Modules 13-17)
- PBR and materials
- Texturing system
- Advanced rendering techniques
- Animation and simulation
- Geometric algorithms
- Specialized effects

### Part IV: Optimization and Integration (Modules 18-20)
- Performance optimization
- Parallelization
- Advanced ray tracing
- Complete rendering systems

---

## Implementation Notes

### Computing Resources
- Any modern computer sufficient
- Multi-core CPU helpful for parallelization
- GPU optional (NVIDIA GPU with CUDA support for GPU acceleration)
- 8GB+ RAM recommended (16GB+ if using CUDA)
- Fast storage (SSD) for faster builds

### Development Approach
- Build incrementally, testing each component
- Start simple, add complexity gradually
- Use version control (Git) from the beginning
- Validate output against reference images
- Debug visually (render normals, depth, etc.)

### Learning Strategy
- Follow the modules in order
- Implement each algorithm completely before moving on
- Test thoroughly with simple scenes first
- Read the reference materials for deeper understanding
- Experiment with variations once basics work

---

*Last Updated: November 2025*

*This tutorial focuses on building 3D graphics systems entirely from first principles without using external graphics libraries. Gain deep understanding by implementing every algorithm yourself in C++. The tutorial philosophy is "build it yourself to truly understand it."*

---

## Acknowledgments

This tutorial draws inspiration from:
- **Academic Courses:** Stanford CS 248, MIT 6.837, Cornell CS 4620, CMU 15-462
- **Books:** "PBR Book" philosophy, "Ray Tracing in One Weekend" approach
- **Philosophy:** "Computer Graphics from Scratch" by Gambetta

### Additional Learning Resources

**Implementatation Guides:**
- Scratchapixel.com - No-API tutorials
- Ray Tracing in One Weekend series
- Tiny renderer project
- Software rasterizer tutorials

**Reference Implementations:**
- PBR Book source code (pbrt)
- SmallPT (99-line path tracer)
- TinyRayTracer
- Software rasterizer examples

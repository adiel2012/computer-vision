# 3D Computer Graphics - Comprehensive Syllabus

## Course Overview
This comprehensive syllabus focuses on classical 3D computer graphics, covering mathematical foundations, geometric modeling, rendering algorithms, and animation techniques. Based on principal textbooks including:
- **Computer Graphics: Principles and Practice (3rd Ed.)** by John F. Hughes, Andries van Dam, Morgan McGuire, David F. Sklar, James D. Foley, Steven K. Feiner, and Kurt Akeley
- **Fundamentals of Computer Graphics (5th Ed.)** by Steve Marschner and Peter Shirley
- **Real-Time Rendering (4th Ed.)** by Tomas Akenine-Möller, Eric Haines, and Naty Hoffman
- **Physically Based Rendering: From Theory to Implementation (3rd Ed.)** by Matt Pharr, Wenzel Jakob, and Greg Humphreys

---

## Module 1: Mathematical Foundations

### Week 1-2: Mathematics for Graphics
- **Vector Mathematics**
  - Vector operations (addition, subtraction, scalar multiplication)
  - Dot product and cross product
  - Vector length and normalization
  - Orthogonal and orthonormal bases
  - Gram-Schmidt orthogonalization

- **Matrix Mathematics**
  - Matrix operations
  - Matrix multiplication
  - Transpose and inverse
  - Determinants
  - Special matrices (identity, diagonal, symmetric)

- **Coordinate Systems and Spaces**
  - Cartesian coordinates
  - Homogeneous coordinates
  - Object space, world space, camera space, screen space
  - Change of basis
  - Coordinate system handedness (left-handed vs. right-handed)

- **Transformations**
  - Translation
  - Rotation (2D and 3D)
  - Scaling (uniform and non-uniform)
  - Shearing
  - Reflection
  - Transformation matrices
  - Composition of transformations
  - Transformation hierarchies

- **Rotation Representations**
  - Euler angles
  - Axis-angle representation
  - Rotation matrices
  - Quaternions
  - Interpolation (SLERP)

**Key References:** Marschner & Shirley Ch. 2, 5-6, Hughes et al. Ch. 5-7

---

## Module 2: 2D Graphics Fundamentals

### Week 3: Rasterization Basics
- **Line Drawing Algorithms**
  - DDA (Digital Differential Analyzer)
  - Bresenham's line algorithm
  - Midpoint line algorithm
  - Anti-aliased lines

- **Circle and Ellipse Drawing**
  - Midpoint circle algorithm
  - Bresenham's circle algorithm
  - Ellipse rasterization

- **Polygon Fill Algorithms**
  - Scan-line polygon fill
  - Flood fill
  - Boundary fill
  - Edge table and active edge table

- **Clipping Algorithms**
  - Cohen-Sutherland line clipping
  - Liang-Barsky line clipping
  - Sutherland-Hodgeman polygon clipping
  - Weiler-Atherton polygon clipping

**Key References:** Marschner & Shirley Ch. 3, 8, Hughes et al. Ch. 3

---

## Module 3: 3D Geometric Modeling

### Week 4-6: Representations and Primitives
- **3D Primitives**
  - Points and vectors in 3D
  - Lines and rays
  - Planes
  - Triangles
  - Spheres, cylinders, cones
  - Bounding volumes (AABB, OBB, spheres)

- **Polygon Meshes**
  - Triangle meshes
  - Quad meshes
  - Mesh data structures
  - Half-edge structure
  - Winged-edge structure
  - Vertex-face structure
  - Mesh topology

- **Parametric Curves**
  - Bézier curves
  - B-spline curves
  - NURBS (Non-Uniform Rational B-Splines)
  - Catmull-Rom splines
  - Hermite curves
  - Curve properties (continuity, convex hull)

- **Parametric Surfaces**
  - Bézier surfaces (patches)
  - B-spline surfaces
  - NURBS surfaces
  - Subdivision surfaces (Catmull-Clark, Loop)
  - Tensor product surfaces
  - Surface properties

- **Implicit Surfaces**
  - Implicit surface definition f(x,y,z) = 0
  - Signed distance functions
  - Metaballs and blobby objects
  - CSG (Constructive Solid Geometry)
  - Boolean operations

- **Procedural Modeling**
  - Fractals
  - L-systems
  - Noise functions (Perlin noise)
  - Terrain generation
  - Procedural textures

**Key References:** Marschner & Shirley Ch. 12, 15, Hughes et al. Ch. 20-23

---

## Module 4: 3D Transformations and Viewing

### Week 7-8: The Graphics Pipeline
- **3D Transformations**
  - Translation matrices
  - Rotation matrices (around x, y, z axes)
  - Scaling matrices
  - General rotation (arbitrary axis)
  - Affine transformations
  - Rigid body transformations
  - Normal transformation

- **Hierarchical Modeling**
  - Scene graphs
  - Transformation hierarchies
  - Parent-child relationships
  - Forward kinematics
  - Matrix stacks

- **Viewing Transformations**
  - Camera positioning (look-at transformation)
  - View matrix construction
  - Synthetic camera model
  - Camera coordinate system

- **Projection Transformations**
  - Orthographic projection
  - Perspective projection
  - Projection matrix derivation
  - Perspective divide
  - Depth and w-component

- **Viewport Transformation**
  - NDC (Normalized Device Coordinates)
  - Screen space mapping
  - Viewport matrix
  - Aspect ratio handling

- **The Complete Pipeline**
  - Model → World → View → Clip → NDC → Screen
  - Homogeneous clipping
  - Perspective-correct interpolation

**Key References:** Marschner & Shirley Ch. 7-8, Hughes et al. Ch. 7-8, Akenine-Möller et al. Ch. 4

---

## Module 5: Visibility and Hidden Surface Removal

### Week 9-10: Culling and Occlusion
- **Back-Face Culling**
  - Normal-based culling
  - View-dependent culling
  - Winding order

- **View Frustum Culling**
  - Frustum planes
  - Bounding volume testing
  - Hierarchical culling
  - Portal culling

- **Occlusion Culling**
  - Occluder and occludee
  - Hardware occlusion queries
  - Potentially visible sets (PVS)
  - Occlusion culling algorithms

- **Hidden Surface Removal Algorithms**
  - Painter's algorithm
  - Z-buffer (depth buffer) algorithm
  - Scanline algorithm
  - Depth sorting (BSP trees)
  - Binary Space Partitioning (BSP)
  - Octrees for culling

**Key References:** Marschner & Shirley Ch. 8, Akenine-Möller et al. Ch. 19

---

## Module 6: Rasterization and Scanline Rendering

### Week 11-12: Triangle Rasterization
- **Triangle Rasterization**
  - Edge equations
  - Barycentric coordinates
  - Scanline rasterization
  - Tile-based rasterization
  - Conservative rasterization

- **Interpolation**
  - Linear interpolation
  - Perspective-correct interpolation
  - Attribute interpolation (colors, normals, texture coordinates)
  - Barycentric interpolation

- **Z-Buffering**
  - Depth buffer algorithm
  - Depth testing
  - Depth precision issues
  - Reversed-Z techniques
  - W-buffering

- **Anti-Aliasing Techniques**
  - Supersampling (SSAA)
  - Multisampling (MSAA)
  - Coverage calculation
  - Edge anti-aliasing
  - Temporal anti-aliasing (TAA)
  - FXAA, SMAA

- **Optimization Techniques**
  - Early-Z rejection
  - Hierarchical Z-buffering
  - Tile-based rendering
  - Deferred shading

**Key References:** Marschner & Shirley Ch. 8, Akenine-Möller et al. Ch. 5, 23

---

## Module 7: Color and Illumination

### Week 13-14: Color Theory and Shading
- **Color Theory**
  - Light and electromagnetic spectrum
  - Human color perception (cones and rods)
  - Tristimulus theory
  - CIE color spaces (XYZ, xyY)
  - RGB color space
  - HSV and HSL
  - Gamma correction and linear color space
  - Color management

- **Light Sources**
  - Point lights (omnidirectional)
  - Directional lights (sun)
  - Spotlights
  - Area lights
  - Ambient light
  - Image-based lighting (IBL)
  - Light attenuation

- **Reflection Models**
  - Diffuse reflection (Lambertian)
  - Specular reflection
  - Ambient reflection
  - Phong reflection model
  - Blinn-Phong model
  - Cook-Torrance model
  - BRDF (Bidirectional Reflectance Distribution Function)

- **Local Illumination Models**
  - Phong shading
  - Flat shading
  - Gouraud shading
  - Per-pixel vs. per-vertex lighting
  - Normal mapping
  - Bump mapping

**Key References:** Marschner & Shirley Ch. 18-20, Hughes et al. Ch. 17, Akenine-Möller et al. Ch. 8-9

---

## Module 8: Ray Tracing Fundamentals

### Week 15-18: Classical Ray Tracing
- **Ray Tracing Basics**
  - Ray representation (origin + direction)
  - Ray-object intersection
  - Primary rays (eye rays)
  - Visibility determination
  - Ray tracing vs. rasterization

- **Ray-Object Intersection**
  - Ray-sphere intersection
  - Ray-plane intersection
  - Ray-triangle intersection (Möller-Trumbore algorithm)
  - Ray-box intersection (slab method)
  - Ray-polygon intersection
  - Barycentric coordinates for triangles
  - Normal computation at intersection

- **Recursive Ray Tracing**
  - Reflection rays
  - Refraction rays (transmission)
  - Fresnel equations
  - Snell's law
  - Total internal reflection
  - Recursive depth limiting

- **Shadow Rays**
  - Hard shadows
  - Shadow testing
  - Multiple light sources
  - Shadow acne and epsilon offsets

- **Acceleration Structures**
  - Bounding Volume Hierarchies (BVH)
    - BVH construction (top-down, bottom-up)
    - Surface Area Heuristic (SAH)
    - BVH traversal
  - Spatial subdivision
    - Uniform grids
    - Octrees
    - k-d trees
  - Traversal algorithms
  - Performance optimization

- **Ray Tracing Optimizations**
  - Early ray termination
  - Mailboxing
  - Packet tracing
  - Coherent ray tracing
  - SIMD optimization

**Key References:** Marschner & Shirley Ch. 4, 13, Pharr et al. Ch. 1, 6-7, Akenine-Möller et al. Ch. 22

---

## Module 9: Advanced Ray Tracing

### Week 19-21: Distribution Ray Tracing and Monte Carlo Methods
- **Distribution Ray Tracing**
  - Soft shadows (area lights)
  - Glossy reflections
  - Translucency
  - Depth of field
  - Motion blur
  - Stochastic sampling

- **Monte Carlo Integration**
  - Monte Carlo estimator
  - Sampling theory
  - Variance reduction
  - Importance sampling
  - Stratified sampling
  - Low-discrepancy sequences (Halton, Sobol)

- **Global Illumination Basics**
  - Rendering equation
  - Direct illumination
  - Indirect illumination
  - Ambient occlusion
  - Radiosity concepts

- **Path Tracing**
  - Path tracing algorithm
  - Russian roulette
  - Multiple importance sampling
  - Light transport notation
  - Convergence and noise

- **Bidirectional Path Tracing**
  - Light paths and eye paths
  - Path connection
  - Multiple importance sampling in BDPT

- **Photon Mapping**
  - Photon tracing
  - Photon storage (kd-tree)
  - Radiance estimation
  - Caustics
  - Final gathering

**Key References:** Pharr et al. Ch. 13-16, Marschner & Shirley Ch. 14, 24

---

## Module 10: Physically Based Rendering (PBR)

### Week 22-24: Material and Light Transport
- **Physically Based BRDF Models**
  - Microfacet theory
  - Cook-Torrance BRDF
  - GGX/Trowbridge-Reitz distribution
  - Fresnel term (Schlick approximation)
  - Geometry/shadowing term
  - Energy conservation
  - Disney principled BRDF

- **Material Models**
  - Dielectrics (glass, water)
  - Conductors (metals)
  - Diffuse materials
  - Subsurface scattering (BSSRDF)
  - Layered materials
  - Anisotropic materials

- **Importance Sampling BRDFs**
  - Cosine-weighted hemisphere sampling
  - BRDF importance sampling
  - Light importance sampling
  - Multiple importance sampling (MIS)

- **Light Transport Theory**
  - Radiometry (radiance, irradiance, flux)
  - Solid angle
  - Rendering equation
  - Light transport paths
  - Measurement equation

- **Volume Rendering**
  - Participating media
  - Absorption, scattering, emission
  - Volumetric path tracing
  - Transmittance
  - Phase functions

**Key References:** Pharr et al. Ch. 5, 8-9, 14, Akenine-Möller et al. Ch. 9

---

## Module 11: Texturing and Surface Detail

### Week 25-26: Texture Mapping
- **Texture Mapping Fundamentals**
  - UV coordinates
  - Texture coordinate generation
  - Planar mapping
  - Cylindrical and spherical mapping
  - Cubic mapping
  - UV unwrapping

- **Texture Filtering**
  - Nearest neighbor
  - Bilinear interpolation
  - Trilinear interpolation
  - Anisotropic filtering
  - Mipmapping
  - Mipmap generation
  - Summed-area tables

- **Advanced Texture Mapping**
  - Normal mapping
  - Bump mapping
  - Displacement mapping
  - Parallax mapping
  - Relief mapping
  - Environment mapping (reflection maps)
  - Cube maps
  - Spherical harmonics

- **Procedural Textures**
  - Noise functions (Perlin, Simplex)
  - Turbulence
  - Marble, wood, clouds
  - Cellular textures (Worley noise)
  - Fractal patterns

- **Texture Compression**
  - Block compression (DXT, BC)
  - ASTC
  - Basis Universal

**Key References:** Marschner & Shirley Ch. 11, Akenine-Möller et al. Ch. 6, Pharr et al. Ch. 10

---

## Module 12: Shadows and Ambient Occlusion

### Week 27-28: Shadow Algorithms
- **Shadow Mapping**
  - Shadow map generation
  - Shadow map lookup
  - Depth comparison
  - Shadow acne and Peter panning
  - Bias techniques
  - Slope-scale bias

- **Shadow Map Filtering**
  - PCF (Percentage Closer Filtering)
  - Soft shadows with PCF
  - PCSS (Percentage Closer Soft Shadows)
  - Variance shadow maps (VSM)
  - Exponential shadow maps (ESM)

- **Cascaded Shadow Maps**
  - Multiple shadow cascades
  - Frustum splitting
  - Blend regions

- **Other Shadow Techniques**
  - Shadow volumes (stencil shadows)
  - Ray traced shadows
  - Screen-space shadows

- **Ambient Occlusion**
  - Ambient occlusion concept
  - Precomputed AO
  - Screen-space ambient occlusion (SSAO)
  - Horizon-based AO (HBAO)
  - Ground-truth AO (ray traced)

**Key References:** Akenine-Möller et al. Ch. 7, Marschner & Shirley Ch. 11

---

## Module 13: Global Illumination for Real-Time

### Week 29-30: Approximations and Precomputation
- **Radiosity**
  - Form factors
  - Radiosity equation
  - Progressive radiosity
  - Limitations and use cases

- **Light Probes and Irradiance Volumes**
  - Reflection probes
  - Irradiance volumes
  - Light probe interpolation
  - Parallax correction

- **Spherical Harmonics**
  - SH basis functions
  - Irradiance encoding
  - SH lighting
  - Prefiltered environment maps

- **Lightmapping**
  - Baked lighting
  - UV layout for lightmaps
  - Seam handling
  - Mixed lighting (baked + dynamic)

- **Voxel-Based Global Illumination**
  - Voxel cone tracing
  - Sparse voxel octrees
  - Cascaded voxel grids

- **Screen-Space Global Illumination**
  - SSGI techniques
  - Screen-space reflections (SSR)
  - Limitations

**Key References:** Akenine-Möller et al. Ch. 11, Marschner & Shirley Ch. 24

---

## Module 14: Animation Fundamentals

### Week 31-33: Keyframe and Procedural Animation
- **Keyframe Animation**
  - Keyframe concept
  - Interpolation between keyframes
  - Linear interpolation
  - Cubic interpolation (Hermite, Bézier)
  - Ease-in, ease-out
  - Animation curves

- **Skeletal Animation**
  - Skeleton hierarchy (bones and joints)
  - Skinning (vertex blending)
  - Linear blend skinning (LBS)
  - Dual quaternion skinning
  - Pose interpolation
  - Animation blending

- **Inverse Kinematics**
  - Forward kinematics vs. inverse kinematics
  - IK algorithms (CCD, FABRIK)
  - IK solvers
  - Constraints

- **Procedural Animation**
  - Physics-based animation
  - Particle systems
  - Cloth simulation
  - Rigid body dynamics
  - Soft body simulation

- **Motion Capture**
  - Motion capture techniques
  - Retargeting
  - Motion editing
  - Motion graphs

- **Morphing and Deformation**
  - Blend shapes (morph targets)
  - Facial animation
  - Deformers (lattice, cage)

**Key References:** Marschner & Shirley Ch. 16-17, Hughes et al. Ch. 16

---

## Module 15: Geometric Processing

### Week 34-35: Mesh Operations
- **Mesh Simplification**
  - Vertex decimation
  - Edge collapse
  - Quadric error metrics
  - Level of detail (LOD)
  - Progressive meshes

- **Mesh Subdivision**
  - Catmull-Clark subdivision
  - Loop subdivision
  - Doo-Sabin subdivision
  - Butterfly subdivision
  - Adaptive subdivision

- **Mesh Smoothing**
  - Laplacian smoothing
  - Taubin smoothing
  - Bilateral filtering
  - Feature-preserving smoothing

- **Mesh Parameterization**
  - UV unwrapping
  - Planar parameterization
  - Conformal mapping
  - Least-squares conformal maps
  - Angle-based flattening

- **Remeshing**
  - Isotropic remeshing
  - Anisotropic remeshing
  - Quadrilateral remeshing

- **Boolean Operations**
  - Union, intersection, difference
  - CSG tree evaluation
  - Mesh intersection algorithms

**Key References:** Marschner & Shirley Ch. 12, Botsch et al. Polygon Mesh Processing

---

## Module 16: Advanced Rendering Techniques

### Week 36-38: Modern Rendering Methods
- **Deferred Shading**
  - G-buffer layout
  - Geometry pass
  - Lighting pass
  - Advantages and disadvantages
  - Deferred vs. forward rendering
  - Light volume optimization

- **Tile-Based Deferred Rendering**
  - Tile culling
  - Light lists per tile
  - Compute shader optimization

- **Clustered Shading**
  - 3D clustering
  - Light assignment
  - Depth slicing

- **Order-Independent Transparency**
  - Depth peeling
  - Weighted blended OIT
  - k-buffer
  - Linked lists

- **Screen-Space Techniques**
  - Screen-space reflections (SSR)
  - Screen-space ambient occlusion (SSAO)
  - Screen-space global illumination
  - Limitations and artifacts

- **Temporal Techniques**
  - Temporal anti-aliasing (TAA)
  - Temporal reprojection
  - History buffers
  - Ghost reduction

**Key References:** Akenine-Möller et al. Ch. 20, 23

---

## Module 17: GPU Architecture and Shader Programming

### Week 39-41: Graphics Hardware and Shaders
- **GPU Architecture**
  - Graphics pipeline stages
  - Programmable vs. fixed function
  - Parallel processing model
  - Warps and wavefronts
  - Memory hierarchy (registers, shared, global)
  - Texture units and samplers

- **Shader Programming**
  - Shader types (vertex, fragment, geometry, compute)
  - Shader inputs and outputs
  - Uniform variables
  - Varying/interpolated variables
  - Built-in functions

- **Vertex Shaders**
  - Vertex transformation
  - Normal transformation
  - Texture coordinate generation
  - Vertex skinning

- **Fragment/Pixel Shaders**
  - Per-pixel lighting
  - Texture sampling
  - Alpha testing and blending
  - Discard operations

- **Geometry Shaders**
  - Primitive generation
  - Point sprites
  - Shadow volume extrusion
  - Use cases and limitations

- **Tessellation Shaders**
  - Hull/control shader
  - Domain/evaluation shader
  - Displacement mapping
  - Adaptive tessellation
  - PN triangles

- **Compute Shaders**
  - General-purpose GPU computing
  - Thread groups
  - Synchronization
  - Use cases (culling, post-processing, physics)

- **Shader Optimization**
  - Instruction cost
  - Branch divergence
  - Texture fetch optimization
  - Precision qualifiers

**Key References:** Akenine-Möller et al. Ch. 3, 20, Hughes et al. Ch. 18

---

## Module 18: Post-Processing Effects

### Week 42-43: Image-Based Effects
- **Bloom and Glow**
  - Bright pass extraction
  - Gaussian blur
  - Bloom composition
  - HDR bloom

- **Depth of Field**
  - Circle of confusion
  - Bokeh simulation
  - Scattered-as-gather
  - Depth-based blur

- **Motion Blur**
  - Per-object motion blur
  - Camera motion blur
  - Velocity buffer
  - Reconstruction filter

- **Tone Mapping**
  - HDR to LDR conversion
  - Reinhard tone mapping
  - Filmic tone mapping
  - ACES tone mapping
  - Exposure control

- **Color Grading**
  - LUT (Look-Up Table)
  - Color correction
  - Color temperature
  - Saturation and contrast

- **Anti-Aliasing Post-Process**
  - FXAA (Fast Approximate AA)
  - SMAA (Subpixel Morphological AA)
  - MLAA (Morphological AA)
  - TAA (Temporal AA)

- **Screen-Space Effects**
  - Vignette
  - Chromatic aberration
  - Film grain
  - Lens distortion

**Key References:** Akenine-Möller et al. Ch. 12, GPU Gems series

---

## Module 19: Non-Photorealistic Rendering

### Week 44-45: Artistic Rendering
- **Edge Detection**
  - Silhouette edges
  - Crease edges
  - Border edges
  - Image-space edge detection (Sobel, Canny)

- **Toon Shading (Cel Shading)**
  - Discrete shading bands
  - Outlines
  - Posterization
  - Ink effects

- **Hatching and Stippling**
  - Line-based rendering
  - Tone mapping to line density
  - Procedural hatching
  - Texture-based hatching

- **Painterly Rendering**
  - Brush stroke simulation
  - Impressionist effects
  - Watercolor simulation

- **Sketch-Based Rendering**
  - Pencil rendering
  - Pen-and-ink rendering
  - Technical illustration style

- **Stylization Techniques**
  - Abstraction
  - Exaggeration
  - Simplification
  - Artistic filters

**Key References:** Marschner & Shirley Ch. 14 (NPR section), NPR-specific texts

---

## Module 20: Real-Time Ray Tracing

### Week 46-48: Hardware-Accelerated Ray Tracing
- **RT Cores and Hardware Acceleration**
  - DirectX Raytracing (DXR)
  - Vulkan Ray Tracing
  - RT core architecture
  - BVH acceleration structures in hardware

- **Hybrid Rendering**
  - Rasterization + ray tracing
  - When to use ray tracing
  - Performance considerations
  - Ray budget management

- **Real-Time Ray Traced Effects**
  - Ray traced reflections
  - Ray traced shadows
  - Ray traced ambient occlusion
  - Ray traced global illumination

- **Denoising**
  - Spatiotemporal denoising
  - SVGF (Spatiotemporal Variance-Guided Filtering)
  - ReSTIR (Reservoir-based Spatio-Temporal Importance Resampling)
  - AI-based denoisers

- **Performance Optimization**
  - Ray coherence
  - Ray sorting
  - Adaptive sampling
  - Variable rate ray tracing

**Key References:** Akenine-Möller et al. Ch. 26, Recent ray tracing papers

---

## Module 21: Advanced Topics in Rendering

### Week 49-51: Specialized Techniques
- **Subsurface Scattering**
  - Diffusion approximation
  - Texture-space diffusion
  - Screen-space subsurface scattering
  - Separable SSS

- **Hair and Fur Rendering**
  - Kajiya-Kay model
  - Marschner model
  - Hair geometry representation
  - Hair shading
  - Deep opacity maps

- **Atmospheric Effects**
  - Fog and aerial perspective
  - Volumetric fog
  - Atmospheric scattering (Rayleigh, Mie)
  - Sky models (Preetham, Hosek-Wilkie)
  - Clouds

- **Water Rendering**
  - Water surface simulation (FFT)
  - Reflection and refraction
  - Caustics
  - Foam and spray
  - Underwater effects

- **Terrain Rendering**
  - Heightmaps
  - LOD and tessellation
  - Texture splatting
  - Parallax occlusion mapping
  - Virtual texturing

- **Vegetation Rendering**
  - Billboard impostors
  - LOD systems
  - Wind animation
  - Subsurface scattering for leaves

**Key References:** GPU Gems series, GPU Pro series, Akenine-Möller et al. specialized chapters

---

## Module 22: Performance and Optimization

### Week 52-53: Rendering Optimization
- **Performance Analysis**
  - Profiling tools
  - GPU profiling
  - Bottleneck identification
  - Frame time analysis

- **CPU Optimization**
  - Culling optimization
  - Batch reduction
  - Instancing
  - Draw call minimization
  - Multi-threading

- **GPU Optimization**
  - Shader optimization
  - Bandwidth reduction
  - Overdraw reduction
  - Occupancy optimization
  - Memory coalescing

- **Level of Detail (LOD)**
  - Mesh LOD
  - Texture LOD
  - Shader LOD
  - Automatic LOD generation
  - Impostor systems

- **Visibility Optimization**
  - Occlusion culling
  - Portal systems
  - Potentially visible sets
  - Hardware occlusion queries

- **Memory Management**
  - Texture streaming
  - Geometry streaming
  - Resource pooling
  - Compression

**Key References:** Akenine-Möller et al. Ch. 18-19, 23

---

## Module 23: Virtual Reality and Advanced Display

### Week 54-55: VR and Special Rendering
- **Virtual Reality Rendering**
  - Stereoscopic rendering
  - Lens distortion correction
  - Foveated rendering
  - Timewarp and spacewarp
  - Latency reduction
  - VR-specific optimizations

- **High Dynamic Range (HDR)**
  - HDR color spaces
  - Rec. 2020, BT.2100
  - PQ and HLG transfer functions
  - HDR tone mapping
  - Display mapping

- **Wide Color Gamut**
  - Color space conversion
  - Gamut mapping
  - Color management

**Key References:** Akenine-Möller et al. Ch. 27, VR-specific publications

---

## Module 24: Production Pipeline and Tools

### Week 56-58: Graphics in Practice
- **Graphics APIs**
  - OpenGL/GLSL
  - DirectX/HLSL
  - Vulkan/SPIR-V
  - Metal
  - WebGL/WebGPU

- **Asset Pipeline**
  - Model formats (OBJ, FBX, glTF)
  - Texture formats
  - Asset conditioning
  - Compression and optimization

- **Rendering Engines**
  - Engine architecture
  - Scene graph management
  - Resource management
  - Renderer abstraction

- **Content Creation Tools**
  - 3D modeling (Blender, Maya, 3ds Max)
  - Texturing (Substance Designer/Painter)
  - Animation tools
  - Export workflows

- **Debugging and Validation**
  - Graphics debuggers (RenderDoc, PIX, Nsight)
  - Shader debugging
  - Performance analysis
  - Validation layers

- **Modern Rendering Techniques Integration**
  - Combining multiple techniques
  - Quality vs. performance trade-offs
  - Scalability
  - Platform considerations

**Key References:** Engine documentation, Tool documentation

---

## Assessment Structure

### Theoretical Components (30%)
- **Weekly Problem Sets (15%)**
  - Mathematical derivations
  - Algorithm analysis
  - Conceptual questions

- **Midterm Examination (8%)**
  - Transformations and viewing
  - Illumination models
  - Ray tracing fundamentals

- **Final Examination (7%)**
  - Advanced rendering techniques
  - Optimization
  - Integration of concepts

### Practical Components (70%)
- **Programming Assignments (50%)**
  1. **2D Rasterization** (6%)
     - Line and circle drawing
     - Polygon fill and clipping

  2. **3D Transformations and Viewing** (8%)
     - Implement transformation pipeline
     - Camera system

  3. **Ray Tracer - Basic** (12%)
     - Ray-object intersection
     - Phong shading
     - Shadows

  4. **Ray Tracer - Advanced** (12%)
     - Reflection and refraction
     - Acceleration structures
     - Distribution ray tracing

  5. **Real-Time Renderer** (12%)
     - OpenGL/DirectX/Vulkan implementation
     - Shader programming
     - Texturing and lighting
     - Shadow mapping

- **Final Project (20%)**
  - Proposal and design (3%)
  - Implementation (12%)
  - Demo and presentation (3%)
  - Written report (2%)

---

## Prerequisites

### Required Background
- **Linear Algebra**: Vectors, matrices, transformations
- **Calculus**: Derivatives, integrals, multivariable calculus
- **Programming**: C++ or similar language proficiency
- **Data Structures**: Trees, spatial structures
- **Algorithms**: Computational geometry basics

### Recommended Background
- Computer architecture basics
- Physics (optics, mechanics)
- Numerical methods

---

## Recommended Textbooks

### Primary Texts

1. **Marschner, Steve & Shirley, Peter.** *Fundamentals of Computer Graphics* (5th Edition), A K Peters/CRC Press, 2021
   - Excellent overall introduction
   - Strong on mathematical foundations
   - Covers both ray tracing and rasterization

2. **Hughes, John F., van Dam, Andries, McGuire, Morgan, et al.** *Computer Graphics: Principles and Practice* (3rd Edition), Addison-Wesley, 2013
   - Comprehensive coverage
   - Classic reference
   - Excellent for fundamentals

3. **Pharr, Matt, Jakob, Wenzel & Humphreys, Greg.** *Physically Based Rendering: From Theory to Implementation* (4th Edition), MIT Press, 2023
   - Definitive ray tracing reference
   - Complete implementation
   - Deep theoretical coverage

4. **Akenine-Möller, Tomas, Haines, Eric & Hoffman, Naty.** *Real-Time Rendering* (4th Edition), A K Peters/CRC Press, 2018
   - Essential for real-time graphics
   - Modern techniques
   - Industry standard

### Supplementary Texts

5. **Shirley, Peter.** *Ray Tracing in One Weekend* series
   - Practical ray tracing introduction
   - Accessible and hands-on

6. **Angel, Edward & Shreiner, Dave.** *Interactive Computer Graphics* (8th Edition), Pearson, 2020
   - OpenGL-focused
   - Good for beginners

7. **Dutre, Philip, Bekaert, Philippe & Bala, Kavita.** *Advanced Global Illumination* (2nd Edition), A K Peters, 2006
   - In-depth global illumination theory

### Online Resources

8. **Scratchapixel** (www.scratchapixel.com)
   - Free online tutorials
   - Covers rendering fundamentals

9. **Learn OpenGL** (learnopengl.com)
   - Modern OpenGL tutorial

10. **Ray Tracing Gems** (free online)
    - Collection of ray tracing techniques

11. **GPU Gems** series (free online from NVIDIA)
    - Advanced rendering techniques

12. **Ke-Sen Huang's Graphics Papers** collection
    - Organized research paper links

---

## Software and Tools

### Required Software

**Graphics APIs (choose one or more):**
- **OpenGL** 4.5+ with GLSL
- **DirectX 12** with HLSL
- **Vulkan** with SPIR-V
- **WebGL** for web-based assignments

**Development Tools:**
- C++ compiler (GCC, Clang, MSVC)
- CMake for build management
- Git for version control

**Libraries:**
- **GLM** (OpenGL Mathematics) - math library
- **GLFW** or **SDL2** - window and input
- **GLAD** or **GLEW** - OpenGL extension loading
- **ImGui** - debug UI
- **stb_image** - image loading
- **tinyobjloader** - model loading

### Recommended Software

**3D Modeling and Content Creation:**
- **Blender** (free, open source)
- **Maya** or **3ds Max** (if available)
- **Substance Designer/Painter**
- **ZBrush**

**Debugging and Profiling:**
- **RenderDoc** - graphics debugger
- **NVIDIA Nsight Graphics**
- **Intel Graphics Performance Analyzers**
- **PIX** (for DirectX)

**Image Editing:**
- **GIMP** or **Photoshop**
- **Krita**

### Development Environment
- Visual Studio, Visual Studio Code, or CLion
- Python for scripting and tools
- Jupyter notebooks for experiments

### Datasets and Resources
- **Stanford 3D Scanning Repository**
- **McGuire Computer Graphics Archive**
- **Poly Haven** - HDR environments and models
- **Physically Based Rendering Scenes**

---

## Learning Outcomes

By the end of this course, students will be able to:

1. **Understand mathematical foundations**:
   - Vectors, matrices, and transformations
   - Coordinate systems and projections
   - Geometric representations

2. **Implement rendering algorithms**:
   - Rasterization pipeline
   - Ray tracing (basic and advanced)
   - Illumination models
   - Shadow algorithms

3. **Work with 3D geometry**:
   - Create and manipulate meshes
   - Parametric curves and surfaces
   - Geometric processing

4. **Program shaders**:
   - Vertex and fragment shaders
   - Compute shaders
   - Shader optimization

5. **Apply texturing techniques**:
   - Texture mapping
   - Normal and displacement mapping
   - Procedural textures

6. **Implement advanced effects**:
   - Shadows and ambient occlusion
   - Reflections and refractions
   - Post-processing effects

7. **Optimize rendering performance**:
   - GPU optimization
   - Culling techniques
   - LOD systems

8. **Use graphics APIs**:
   - OpenGL, DirectX, or Vulkan
   - Modern graphics programming patterns

9. **Build complete rendering systems**:
   - Integration of multiple techniques
   - Real-time and offline renderers

10. **Understand production pipelines**:
    - Asset workflows
    - Tools and debugging

---

## Course Schedule Overview

### Part I: Foundations (Weeks 1-14)
- Mathematics and transformations
- 2D graphics
- 3D modeling
- Viewing and projection
- Visibility
- Rasterization
- Color and illumination

### Part II: Ray Tracing (Weeks 15-24)
- Ray tracing fundamentals
- Acceleration structures
- Distribution ray tracing
- Monte Carlo methods
- Path tracing
- Physically based rendering
- Materials and BRDFs

### Part III: Real-Time Rendering (Weeks 25-43)
- Texturing
- Shadows
- Global illumination approximations
- Animation
- Geometric processing
- Advanced rendering techniques
- GPU programming
- Post-processing

### Part IV: Advanced Topics (Weeks 44-58)
- Non-photorealistic rendering
- Real-time ray tracing
- Specialized rendering (hair, water, atmosphere)
- Performance optimization
- VR and HDR
- Production pipeline

---

## Weekly Time Commitment
- **Lectures**: 3 hours
- **Lab/Tutorial sessions**: 2 hours
- **Problem sets**: 3-5 hours
- **Reading**: 3-4 hours
- **Programming assignments**: 8-15 hours
- **Total**: 19-29 hours per week

---

## Project Ideas

### Ray Tracing Projects
- Photorealistic scene renderer
- Bidirectional path tracer
- Photon mapping implementation
- Volumetric rendering
- Caustics simulation

### Real-Time Projects
- Game engine renderer
- Deferred rendering system
- Physically-based real-time renderer
- VR rendering system
- Procedural world generator

### Specialized Projects
- Non-photorealistic renderer
- Cloth simulation and rendering
- Fluid simulation and rendering
- Hair rendering system
- Real-time global illumination

---

## Additional Notes

### Computing Resources
- GPU required (NVIDIA or AMD)
- Minimum 4GB VRAM recommended
- 16GB+ system RAM
- Multi-core CPU helpful

### Recommended Hardware
- Modern GPU (RTX series for ray tracing assignments)
- Multiple monitors helpful
- Good color-calibrated display

---

*Last Updated: November 2025*

*This syllabus covers classical and modern computer graphics techniques, including both offline (ray tracing) and real-time (rasterization) rendering. The course emphasizes both theoretical understanding and practical implementation.*

---

## Acknowledgments
This syllabus draws inspiration from courses at:
- Stanford University (CS 148, CS 348B)
- MIT (6.837)
- UC Berkeley (CS 184/284A)
- Cornell University (CS 4620/4621)
- University of Washington (CSE 457/557)
- Carnegie Mellon University (15-462/662)

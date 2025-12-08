# Tutorial Roadmap - 3D Computer Graphics

A visual guide to navigating the 20-chapter tutorial.

## 🗺️ Complete Learning Path

```
START HERE
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  PART I: FOUNDATIONS (Chapters 1-7)                             │
│  Build a Software Rasterizer                                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─→ Ch 1: Math Foundations ────────────→ Vec3, Mat4, Quaternions
    │
    ├─→ Ch 2: Framebuffer & 2D ────────────→ Bresenham, DDA
    │
    ├─→ Ch 3: 3D Geometry ─────────────────→ Meshes, Primitives, Curves
    │
    ├─→ Ch 4: Transformations ─────────────→ Model/View/Projection
    │
    ├─→ Ch 5: Visibility ──────────────────→ Z-buffer, Culling, BSP
    │
    ├─→ Ch 6: Triangle Rasterization ──────→ Barycentric, Edge Functions
    │
    └─→ Ch 7: Shading & Lighting ──────────→ Phong, Blinn-Phong
         │
         ✓ MILESTONE: Can render 3D meshes with lighting!
         │
┌─────────────────────────────────────────────────────────────────┐
│  PART II: RAY TRACING (Chapters 8-12)                           │
│  Build a Path Tracer                                             │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─→ Ch 8: Core Ray Tracing ────────────→ Intersections, Shadows, Reflections
    │
    ├─→ Ch 9: Acceleration ────────────────→ BVH, k-d Tree, Octree
    │
    ├─→ Ch 10: Distribution RT ────────────→ Soft Shadows, DOF, Motion Blur
    │
    ├─→ Ch 11: Path Tracing ───────────────→ Monte Carlo, Global Illumination
    │
    └─→ Ch 12: Physically Based ───────────→ Cook-Torrance, Microfacets, PBR
         │
         ✓ MILESTONE: Can render photorealistic images!
         │
┌─────────────────────────────────────────────────────────────────┐
│  PART III: ADVANCED FEATURES (Chapters 13-17)                   │
│  Production-Quality Rendering                                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─→ Ch 13: Texturing ──────────────────→ UV Mapping, Mipmaps, Filtering
    │
    ├─→ Ch 14: Advanced Rendering ─────────→ AO, SSS, Volumetrics, Bloom
    │
    ├─→ Ch 15: Animation ──────────────────→ Keyframes, Physics, Particles
    │
    ├─→ Ch 16: Geometric Algorithms ───────→ Subdivision, Simplification
    │
    └─→ Ch 17: Advanced Topics ────────────→ Photon Mapping, BDPT, MLT
         │
         ✓ MILESTONE: Production-ready features!
         │
┌─────────────────────────────────────────────────────────────────┐
│  PART IV: OPTIMIZATION & PROJECTS (Chapters 18-20)              │
│  Make It Fast & Complete                                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─→ Ch 18: Optimization ───────────────→ Multi-threading, SIMD, CUDA
    │
    ├─→ Ch 19: Advanced RT Topics ─────────→ Real-time, Denoising, Hybrid
    │
    └─→ Ch 20: Integration & Projects ─────→ Complete Systems, 12 Projects
         │
         ✓ MILESTONE: Complete rendering systems!
         │
    MASTERY! 🎓
```

## 📚 Chapter Dependencies

### Independent Modules (Can Study in Any Order)
- Chapter 15 (Animation) - needs Ch 1 only
- Chapter 16 (Geometric Algorithms) - needs Ch 1, 3 only

### Linear Dependencies (Must Follow Order)

**Rasterization Track:**
```
Ch 1 → Ch 2 → Ch 3 → Ch 4 → Ch 5 → Ch 6 → Ch 7
```

**Ray Tracing Track:**
```
Ch 1 → Ch 8 → Ch 9 → Ch 10 → Ch 11 → Ch 12 → Ch 13 → Ch 14
```

**Optimization Track:**
```
Ch 8 or Ch 6 → Ch 18 → Ch 19
```

## 🎯 Learning Tracks

### Track A: Rasterization Focus (Game Development)
Perfect for understanding how game engines work.

```
Week 1-2:   Ch 1  (Math)
Week 3-4:   Ch 2  (2D Rasterization)
Week 5-6:   Ch 3  (3D Geometry)
Week 7-8:   Ch 4  (Transformations)
Week 9-10:  Ch 5  (Visibility)
Week 11-13: Ch 6  (Triangle Rasterization)
Week 14-16: Ch 7  (Shading)
Week 17-18: Ch 13 (Texturing)
Week 19-20: Ch 18 (Optimization)

PROJECT: Build a software rasterizer with textures and lighting
```

### Track B: Ray Tracing Focus (Film/VFX)
Perfect for understanding production renderers like RenderMan.

```
Week 1-2:   Ch 1  (Math)
Week 3-5:   Ch 8  (Core Ray Tracing)
Week 6-8:   Ch 9  (Acceleration)
Week 9-11:  Ch 10 (Distribution RT)
Week 12-15: Ch 11 (Path Tracing)
Week 16-19: Ch 12 (PBR)
Week 20-22: Ch 13 (Texturing)
Week 23-25: Ch 14 (Advanced Rendering)
Week 26-28: Ch 18 (GPU Optimization)

PROJECT: Build a GPU-accelerated path tracer
```

### Track C: Complete Mastery (Academic/Research)
Full understanding of all rendering techniques.

```
Month 1:  Ch 1-4   (Foundations)
Month 2:  Ch 5-7   (Rasterization Complete)
Month 3:  Ch 8-9   (Ray Tracing Basics)
Month 4:  Ch 10-11 (Distribution + Path Tracing)
Month 5:  Ch 12-14 (PBR + Advanced)
Month 6:  Ch 15-17 (Animation + Advanced Topics)
Month 7:  Ch 18-19 (Optimization)
Month 8:  Ch 20    (Projects)

PROJECT: Hybrid renderer with all features
```

### Track D: Speed Run (Quick Understanding)
Hit the highlights in 4 weeks.

```
Week 1: Ch 1  (Math) + Ch 2 (2D Basics)
Week 2: Ch 6  (Rasterization) + Ch 7 (Shading)
Week 3: Ch 8  (Ray Tracing) + Ch 11 (Path Tracing)
Week 4: Ch 12 (PBR) + Ch 20 (Review + Projects)

PROJECT: Simple path tracer
```

## 🔑 Key Concepts by Chapter

| Chapter | Core Algorithm | Key Equation | Implementation |
|---------|---------------|--------------|----------------|
| 1 | Matrix Math | `M * v` | Mat4 class |
| 2 | Bresenham | `y = mx + b` | Line drawing |
| 3 | Bézier Curves | `B(t) = Σ bᵢBᵢ(t)` | De Casteljau |
| 4 | MVP Pipeline | `P * V * M * v` | Transform chain |
| 5 | Z-Buffer | `if (z < zbuf[x,y])` | Depth testing |
| 6 | Rasterization | Barycentric | Edge functions |
| 7 | Phong | `I = Iₐ + Id + Is` | Shading |
| 8 | Ray Tracing | `r(t) = o + td` | Intersection |
| 9 | BVH | SAH split | Tree traversal |
| 10 | Stochastic | Monte Carlo | Random sampling |
| 11 | Path Tracing | Rendering Eq | Russian roulette |
| 12 | PBR | Cook-Torrance | GGX NDF |
| 13 | Texturing | Bilinear | Mipmapping |
| 14 | AO/SSS | Ray marching | Volume rendering |
| 15 | Physics | F = ma | Verlet integration |
| 16 | Subdivision | Catmull-Clark | Mesh refinement |
| 17 | Photon Map | k-d tree | Radiance estimate |
| 18 | CUDA | Parallel | GPU kernels |
| 19 | Denoising | Wavelet | Filter |
| 20 | Integration | Scene graph | Complete system |

## 💡 Difficulty Progression

```
EASY          │ Ch 1, 2, 3
              │ Foundation concepts
              │
MODERATE      │ Ch 4, 5, 6, 7, 8
              │ Core rendering algorithms
              │
CHALLENGING   │ Ch 9, 10, 11, 13
              │ Advanced algorithms
              │
ADVANCED      │ Ch 12, 14, 17, 18, 19
              │ Production techniques
              │
INTEGRATION   │ Ch 15, 16, 20
              │ Putting it all together
```

## 🎓 Skill Development

### After Chapter 7 (Rasterization)
**You can:**
- Implement software rasterizer
- Understand GPU pipeline
- Write simple 3D engines
- Debug rendering issues

**Jobs:** Junior graphics programmer, technical artist

### After Chapter 12 (Ray Tracing)
**You can:**
- Build path tracers
- Understand PBR
- Implement offline renderers
- Read SIGGRAPH papers

**Jobs:** Rendering engineer, VFX TD

### After Chapter 17 (Advanced)
**You can:**
- Implement cutting-edge techniques
- Design rendering systems
- Optimize for production
- Research new methods

**Jobs:** Senior rendering engineer, researcher

### After Chapter 20 (Complete)
**You can:**
- Build complete rendering systems
- Lead graphics teams
- Publish research
- Contribute to production renderers

**Jobs:** Lead engineer, principal engineer, researcher

## 📊 Estimated Time Investment

| Track | Duration | Effort | Outcome |
|-------|----------|--------|---------|
| Speed Run | 4 weeks | 10 hrs/week | Basic understanding |
| Rasterization | 10 weeks | 10 hrs/week | Software rasterizer |
| Ray Tracing | 14 weeks | 12 hrs/week | Path tracer |
| Complete | 24 weeks | 15 hrs/week | Full mastery |
| Mastery + Projects | 40 weeks | 20 hrs/week | Production-ready |

## 🚀 Quick Navigation

### By Topic
- **Math**: Ch 1
- **2D Graphics**: Ch 2
- **3D Basics**: Ch 3, 4, 5
- **Rasterization**: Ch 6, 7
- **Ray Tracing**: Ch 8, 9, 10, 11
- **Materials**: Ch 7, 12, 13
- **Effects**: Ch 14, 17
- **Motion**: Ch 15
- **Geometry**: Ch 16
- **Performance**: Ch 18, 19
- **Projects**: Ch 20

### By Difficulty
- **Beginner**: 1, 2, 3, 8
- **Intermediate**: 4, 5, 6, 7, 9, 10
- **Advanced**: 11, 12, 13, 14, 15, 16
- **Expert**: 17, 18, 19, 20

### By Application
- **Games**: 1, 2, 4, 5, 6, 7, 13, 15, 18
- **Film/VFX**: 1, 8, 9, 10, 11, 12, 14, 17
- **Research**: All chapters
- **Education**: 1-12, 20

## 📖 Chapter Reference

| # | Title | Colab Link |
|---|-------|------------|
| 1 | Mathematical Foundations | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_01_math_foundations.ipynb) |
| 2 | Framebuffer and 2D | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_02_framebuffer_2d.ipynb) |
| 3 | 3D Geometry | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_03_3d_geometry.ipynb) |
| 4 | Transformations | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_04_transformations.ipynb) |
| 5 | Visibility | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_05_visibility_and_hidden_surface_removal.ipynb) |
| 6 | Triangle Rasterization | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_06_triangle_rasterization.ipynb) |
| 7 | Shading | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_07_shading_and_illumination.ipynb) |
| 8 | Ray Tracing Core | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_08_ray_tracing.ipynb) |
| 9 | Acceleration | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_09_acceleration_structures.ipynb) |
| 10 | Distribution RT | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_10_distribution_ray_tracing.ipynb) |
| 11 | Path Tracing | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_11_path_tracing.ipynb) |
| 12 | PBR | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_12_physically_based_rendering.ipynb) |
| 13 | Texturing | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_13_texturing.ipynb) |
| 14 | Advanced Rendering | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_14_advanced_rendering.ipynb) |
| 15 | Animation | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_15_animation_and_simulation.ipynb) |
| 16 | Geometric Algorithms | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_16_geometric_algorithms.ipynb) |
| 17 | Advanced Topics | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_17_advanced_topics.ipynb) |
| 18 | Optimization | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_18_optimization_and_parallelization.ipynb) |
| 19 | Advanced RT | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_19_advanced_ray_tracing_topics.ipynb) |
| 20 | Projects | [Open](https://colab.research.google.com/github/adiel2012/computer-vision/blob/main/chapter_20_integration_and_projects.ipynb) |

---

**Choose your path and start learning!** 🚀

All roads lead to mastery - pick the one that excites you most!

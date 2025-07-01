use crate::{
    pipeline::Pipeline,
    traits::{ImagePreprocessor, ContourExtractor, HoleDetector, ShapePostProcessor},
    algorithms::{
        ImageprocContourExtractor, 
        ContainmentHoleDetector,
        ThresholdPreprocessor,
        DouglasPeuckerSimplifier,
        VisvalingamWhyattSimplifier,
        ChaikinSmoothingProcessor,
        GeometryValidator,
        ConvexHullProcessor,
    },
};

/// Builder for creating processing pipelines with a fluent API
pub struct PipelineBuilder {
    preprocessors: Vec<Box<dyn ImagePreprocessor>>,
    contour_extractor: Option<Box<dyn ContourExtractor>>,
    hole_detector: Option<Box<dyn HoleDetector>>,
    postprocessors: Vec<Box<dyn ShapePostProcessor>>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            preprocessors: Vec::new(),
            contour_extractor: None,
            hole_detector: None,
            postprocessors: Vec::new(),
        }
    }

    /// Add a preprocessor to the pipeline
    pub fn add_preprocessor<P>(mut self, preprocessor: P) -> Self
    where
        P: ImagePreprocessor + 'static,
    {
        self.preprocessors.push(Box::new(preprocessor));
        self
    }

    /// Set the contour extractor (replaces any existing one)
    pub fn set_contour_extractor<E>(mut self, extractor: E) -> Self
    where
        E: ContourExtractor + 'static,
    {
        self.contour_extractor = Some(Box::new(extractor));
        self
    }

    /// Set the hole detector (replaces any existing one)
    pub fn set_hole_detector<H>(mut self, detector: H) -> Self
    where
        H: HoleDetector + 'static,
    {
        self.hole_detector = Some(Box::new(detector));
        self
    }

    /// Add a post-processor to the pipeline
    pub fn add_postprocessor<P>(mut self, postprocessor: P) -> Self
    where
        P: ShapePostProcessor + 'static,
    {
        self.postprocessors.push(Box::new(postprocessor));
        self
    }

    /// Add Douglas-Peucker simplification as a post-processing step
    pub fn with_simplification(self, tolerance: f32) -> Self {
        let simplifier = SimplificationProcessor::new(tolerance, SimplificationMethod::DouglasPeucker);
        self.add_postprocessor(simplifier)
    }

    /// Add Visvalingam-Whyatt simplification as a post-processing step
    pub fn with_vw_simplification(self, tolerance: f32) -> Self {
        let simplifier = SimplificationProcessor::new(tolerance, SimplificationMethod::VisvalingamWhyatt);
        self.add_postprocessor(simplifier)
    }

    /// Add Chaikin smoothing as a post-processing step
    pub fn with_chaikin_smoothing(self, iterations: usize) -> Self {
        let smoother = ChaikinSmoothingProcessor { iterations };
        self.add_postprocessor(smoother)
    }

    /// Add geometry validation as a post-processing step
    pub fn with_validation(self) -> Self {
        self.add_postprocessor(GeometryValidator)
    }

    /// Convert shapes to convex hulls
    pub fn with_convex_hull(self) -> Self {
        self.add_postprocessor(ConvexHullProcessor)
    }

    /// Build the pipeline with default components if not specified
    pub fn build(self) -> Pipeline {
        let contour_extractor = self.contour_extractor
            .unwrap_or_else(|| Box::new(ImageprocContourExtractor));
        
        let hole_detector = self.hole_detector
            .unwrap_or_else(|| Box::new(ContainmentHoleDetector));

        Pipeline::new(
            self.preprocessors,
            contour_extractor,
            hole_detector,
            self.postprocessors,
        )
    }

    /// Build a simple pipeline with basic threshold preprocessing
    pub fn build_simple(threshold: u8) -> Pipeline {
        Self::new()
            .add_preprocessor(ThresholdPreprocessor { threshold })
            .build()
    }

    /// Build a pipeline with hole detection
    pub fn build_with_holes(threshold: u8) -> Pipeline {
        Self::new()
            .add_preprocessor(ThresholdPreprocessor { threshold })
            .set_hole_detector(ContainmentHoleDetector)
            .build()
    }

    /// Build a pipeline with simplification
    pub fn build_with_simplification(threshold: u8, tolerance: f32) -> Pipeline {
        Self::new()
            .add_preprocessor(ThresholdPreprocessor { threshold })
            .with_simplification(tolerance)
            .build()
    }

    /// Build a comprehensive pipeline with multiple processing steps
    pub fn build_comprehensive(threshold: u8, tolerance: f32, smoothing_iterations: usize) -> Pipeline {
        Self::new()
            .add_preprocessor(ThresholdPreprocessor { threshold })
            .set_hole_detector(ContainmentHoleDetector)
            .with_chaikin_smoothing(smoothing_iterations)
            .with_simplification(tolerance)
            .with_validation()
            .build()
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplification method enum
#[derive(Debug, Clone)]
enum SimplificationMethod {
    DouglasPeucker,
    VisvalingamWhyatt,
}

/// Wrapper to use ShapeSimplifier as ShapePostProcessor
struct SimplificationProcessor {
    tolerance: f32,
    method: SimplificationMethod,
}

impl SimplificationProcessor {
    fn new(tolerance: f32, method: SimplificationMethod) -> Self {
        Self {
            tolerance,
            method,
        }
    }
}

impl ShapePostProcessor for SimplificationProcessor {
    fn process(&self, shapes: &mut [crate::types::ComplexShape]) -> crate::error::Result<()> {
        use crate::traits::ShapeSimplifier;
        
        match self.method {
            SimplificationMethod::DouglasPeucker => {
                let simplifier = DouglasPeuckerSimplifier;
                simplifier.simplify(shapes, self.tolerance)
            }
            SimplificationMethod::VisvalingamWhyatt => {
                let simplifier = VisvalingamWhyattSimplifier;
                simplifier.simplify(shapes, self.tolerance)
            }
        }
    }
} 
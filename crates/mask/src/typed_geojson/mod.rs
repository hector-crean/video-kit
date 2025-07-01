use std::marker::PhantomData;
use serde::{Deserialize, Serialize};
use geojson::{FeatureCollection, GeoJson, Geometry, JsonObject};
use ts_rs::TS;
use schemars::JsonSchema;

/// Properties for mask outline features
#[derive(Serialize, Deserialize, Debug, Clone, TS, JsonSchema)]
#[ts(export)]
#[schemars(description = "Properties for mask outline features")]
pub struct MaskOutlineProperties {
    #[schemars(description = "Unique identifier for the shape")]
    pub id: u32,
    #[schemars(description = "Area of the shape in square pixels")]
    pub area: f64,
    #[schemars(description = "Whether the shape contains holes")]
    pub has_holes: bool,
    #[schemars(description = "Number of holes in the shape")]
    pub hole_count: usize,
    #[schemars(description = "Perimeter of the shape in pixels")]
    pub perimeter: f64,
}

/// Type alias for mask outline GeoJSON
pub type MaskGeoJson = TypedGeoJson<MaskOutlineProperties>;

/// A typed GeoJSON Feature that is generic over its properties.
#[derive(Serialize, Deserialize, Debug)]
pub struct TypedFeature<P> {
    #[serde(flatten)]
    pub feature: geojson::Feature,
    #[serde(skip)]
    _properties: PhantomData<P>,
}


impl<P> TypedFeature<P>
where
    for<'de> P: Serialize + Deserialize<'de>,
{
    /// Creates a new TypedFeature.
    pub fn new(geometry: Option<Geometry>, properties: P) -> Self {
        let feature = geojson::Feature {
            bbox: None,
            geometry,
            id: None,
            properties: serde_json::to_value(properties).ok().and_then(|v| v.as_object().cloned()),
            foreign_members: None,
        };
        Self {
            feature,
            _properties: PhantomData,
        }
    }

    /// Tries to access the typed properties of the feature.
    pub fn properties(&self) -> Option<P> {
        self.feature.properties.as_ref().and_then(|p| {
            serde_json::from_value(serde_json::Value::Object(p.clone())).ok()
        })
    }
}


#[derive(Serialize, Deserialize, Debug)]
pub struct TypedFeatureCollection<P> {
    pub bbox: Option<Vec<f64>>,
    pub features: Vec<TypedFeature<P>>,
    pub foreign_members: Option<JsonObject>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum TypedGeoJson<P> {
    Geometry(Geometry),
    Feature(TypedFeature<P>),
    FeatureCollection(TypedFeatureCollection<P>),
}

impl<P> TypedGeoJson<P> {
    /// Get the underlying FeatureCollection if this is a FeatureCollection variant
    pub fn as_feature_collection(&self) -> Option<&TypedFeatureCollection<P>> {
        match self {
            TypedGeoJson::FeatureCollection(fc) => Some(fc),
            _ => None,
        }
    }
    
    /// Convert to FeatureCollection, consuming self
    pub fn into_feature_collection(self) -> Option<TypedFeatureCollection<P>> {
        match self {
            TypedGeoJson::FeatureCollection(fc) => Some(fc),
            _ => None,
        }
    }
}

impl<P> TypedFeatureCollection<P> {
    /// Get the number of features
    pub fn len(&self) -> usize {
        self.features.len()
    }
    
    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
    
    /// Get features as a slice
    pub fn features(&self) -> &[TypedFeature<P>] {
        &self.features
    }
}

impl MaskGeoJson {
    /// Get features that have holes
    pub fn features_with_holes(&self) -> Vec<&TypedFeature<MaskOutlineProperties>> {
        if let Some(fc) = self.as_feature_collection() {
            fc.features.iter()
                .filter(|feature| {
                    feature.properties()
                        .map(|props| props.has_holes)
                        .unwrap_or(false)
                })
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get features by area range
    pub fn features_by_area_range(&self, min_area: f64, max_area: f64) -> Vec<&TypedFeature<MaskOutlineProperties>> {
        if let Some(fc) = self.as_feature_collection() {
            fc.features.iter()
                .filter(|feature| {
                    if let Some(props) = feature.properties() {
                        props.area >= min_area && props.area <= max_area
                    } else {
                        false
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get the largest feature by area
    pub fn largest_feature(&self) -> Option<&TypedFeature<MaskOutlineProperties>> {
        if let Some(fc) = self.as_feature_collection() {
            fc.features.iter()
                .max_by(|a, b| {
                    let area_a = a.properties().map(|p| p.area).unwrap_or(0.0);
                    let area_b = b.properties().map(|p| p.area).unwrap_or(0.0);
                    area_a.partial_cmp(&area_b).unwrap_or(std::cmp::Ordering::Equal)
                })
        } else {
            None
        }
    }
    
    /// Get metadata from foreign members
    pub fn image_dimensions(&self) -> Option<(u32, u32)> {
        if let Some(fc) = self.as_feature_collection() {
            if let Some(ref foreign) = fc.foreign_members {
                let width = foreign.get("image_width")?.as_u64()? as u32;
                let height = foreign.get("image_height")?.as_u64()? as u32;
                Some((width, height))
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Get shape count from foreign members
    pub fn shape_count(&self) -> Option<usize> {
        if let Some(fc) = self.as_feature_collection() {
            fc.foreign_members.as_ref()
                .and_then(|foreign| foreign.get("shape_count"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        } else {
            None
        }
    }
}





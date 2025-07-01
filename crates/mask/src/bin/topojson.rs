use geojson::GeoJson;
use topojson::{TopoJson, to_geojson};
use geo::{BoundingRect, Centroid, Area, SimplifyVw};
use geo_types::{Geometry as GeoGeometry, Polygon, MultiPolygon, LineString, Coord};





// Coordinate precision for geographic coordinates (decimal degrees)
const COORDINATE_PRECISION: u32 = 3; // 3 decimal places (~110m precision, good for country-level mapping)

// Simplification and filterixng parameters
const SIMPLIFY_TOLERANCE: f64 = 0.01; // Simplification tolerance in degrees
const MIN_AREA_THRESHOLD: f64 = 0.1; // Minimum area in square degrees to keep a feature
const MIN_RING_AREA: f64 = 0.01; // Minimum area for polygon rings (removes small islands)
const MAX_FEATURES: Option<usize> = None; // No limit - keep all specified countries

// Countries to include (specific regions only)
const INCLUDED_COUNTRIES: &[&str] = &[
    // North America
    "Canada",
    "United States of America",
    "United States",
    
    // Europe
    "Netherlands",
    "Italy", 
    "Czech Republic",
    "Czechia",
    "Norway",
    "Estonia",
    "Denmark",
    "Poland",
    "Spain",
    "Germany",
    "Lithuania",
    "United Kingdom",
    "France",
    
    // Australia
    "Australia",
    
    // South America
    "Brazil",
    "Peru",
    
    // Africa
    "South Africa",
    
    // East Asia
    "Japan",
    "Singapore", 
    "Taiwan",
];

fn main() {
    let topojson_str = include_str!("./world-topo.json");
    // Parse to Topology:
    let topo = topojson_str.parse::<TopoJson>().unwrap();

    // Conversion to GeoJson FeatureCollection for the "units" object:
    let feature_collection = match topo {
        TopoJson::Topology(t) => {
            to_geojson(&t, &String::from("units")).expect("Unable to convert TopoJSON to GeoJSON")
        }
        _ => unimplemented!(),
    };

    // Convert to our geojson crate version via JSON
    let fc_json = serde_json::to_string(&feature_collection).unwrap();
    let mut our_fc: geojson::FeatureCollection = serde_json::from_str(&fc_json).unwrap();

    println!("Original features: {}", our_fc.features.len());

    // Filter to only include specified countries
    our_fc.features.retain(|feature| {
        if let Some(properties) = &feature.properties {
            if let Some(name) = properties.get("name") {
                if let Some(name_str) = name.as_str() {
                    return INCLUDED_COUNTRIES.contains(&name_str);
                }
            }
          
        }
        false // Exclude if no matching name found
    });

    println!("After filtering to included countries: {}", our_fc.features.len());

    // Add bbox and centroid to each feature
    for feature in &mut our_fc.features {
        add_bbox_and_centroid(feature);
    }

    // Filter by area and simplify geometries
    let mut filtered_features = Vec::new();
    for mut feature in our_fc.features {
        if let Some(mut simplified_feature) = filter_and_simplify_feature(feature) {
            // Round all coordinates to reduce precision
            round_geometry_coordinates(&mut simplified_feature);
            filtered_features.push(simplified_feature);
        }
    }

    // Sort by area (largest first) and limit number of features if specified
    filtered_features.sort_by(|a, b| {
        let area_a = get_feature_area(a);
        let area_b = get_feature_area(b);
        area_b.partial_cmp(&area_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(max_count) = MAX_FEATURES {
        filtered_features.truncate(max_count);
    }

    println!("Final features after filtering and simplification: {}", filtered_features.len());

    our_fc.features = filtered_features;

    // Convert to final GeoJSON
    let final_geojson = GeoJson::FeatureCollection(our_fc);
    
    std::fs::write("world-simplified-geojson.json", final_geojson.to_string()).unwrap();
    println!("Generated optimized world-simplified-geojson.json for web SVG rendering");
}

fn add_bbox_and_centroid(feature: &mut geojson::Feature) {
    if let Some(ref geometry) = feature.geometry {
        // Calculate bounding box from geographic coordinates
        let mut min_lon = f64::INFINITY;
        let mut min_lat = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let mut all_coords = Vec::new();
        
        // Extract all coordinates from the geometry
        match &geometry.value {
            geojson::Value::Point(coords) => {
                all_coords.push([coords[0], coords[1]]);
            }
            geojson::Value::LineString(coords) => {
                all_coords.extend(coords.iter().map(|c| [c[0], c[1]]));
            }
            geojson::Value::Polygon(rings) => {
                for ring in rings {
                    all_coords.extend(ring.iter().map(|c| [c[0], c[1]]));
                }
            }
            geojson::Value::MultiPoint(coords) => {
                all_coords.extend(coords.iter().map(|c| [c[0], c[1]]));
            }
            geojson::Value::MultiLineString(lines) => {
                for line in lines {
                    all_coords.extend(line.iter().map(|c| [c[0], c[1]]));
                }
            }
            geojson::Value::MultiPolygon(polygons) => {
                for polygon in polygons {
                    for ring in polygon {
                        all_coords.extend(ring.iter().map(|c| [c[0], c[1]]));
                    }
                }
            }
            geojson::Value::GeometryCollection(_) => {
                // Skip geometry collections for now
                return;
            }
        }
        
        // Calculate bounds and centroid
        if !all_coords.is_empty() {
            for coord in &all_coords {
                min_lon = min_lon.min(coord[0]);
                min_lat = min_lat.min(coord[1]);
                max_lon = max_lon.max(coord[0]);
                max_lat = max_lat.max(coord[1]);
            }
            
            // Set bounding box
            let bbox = vec![min_lon, min_lat, max_lon, max_lat];
            feature.bbox = Some(bbox);
            
            // Calculate centroid as center of bounding box
            let centroid_lon = (min_lon + max_lon) / 2.0;
            let centroid_lat = (min_lat + max_lat) / 2.0;
            
            // Add centroid to properties
            if feature.properties.is_none() {
                feature.properties = Some(serde_json::Map::new());
            }
            
            if let Some(ref mut properties) = feature.properties {
                properties.insert(
                    "centroid".to_string(),
                    serde_json::Value::Array(vec![
                        serde_json::Value::from(round_coordinate(centroid_lon)),
                        serde_json::Value::from(round_coordinate(centroid_lat))
                    ])
                );
                
                // Add coordinate system info to properties
                properties.insert(
                    "projection".to_string(),
                    serde_json::Value::String("EPSG:4326".to_string())
                );
                
                properties.insert(
                    "coordinate_system".to_string(),
                    serde_json::Value::String("WGS84".to_string())
                );
                
                properties.insert(
                    "units".to_string(),
                    serde_json::Value::String("degrees".to_string())
                );
            }
        }
    }
}

/// Round coordinates to specified precision
fn round_coordinate(coord: f64) -> f64 {
    if COORDINATE_PRECISION == 0 {
        coord.round()
    } else {
        let multiplier = 10_f64.powi(COORDINATE_PRECISION as i32);
        (coord * multiplier).round() / multiplier
    }
}

/// Filter feature by area and simplify its geometry
fn filter_and_simplify_feature(mut feature: geojson::Feature) -> Option<geojson::Feature> {
    if let Some(ref mut geometry) = feature.geometry {
        match &mut geometry.value {
            geojson::Value::Polygon(rings) => {
                let simplified_rings = simplify_polygon_rings(rings);
                if simplified_rings.is_empty() {
                    return None; // Skip if no rings remain after filtering
                }
                geometry.value = geojson::Value::Polygon(simplified_rings);
            }
            geojson::Value::MultiPolygon(polygons) => {
                let mut simplified_polygons = Vec::new();
                for polygon in polygons {
                    let simplified_rings = simplify_polygon_rings(polygon);
                    if !simplified_rings.is_empty() {
                        simplified_polygons.push(simplified_rings);
                    }
                }
                if simplified_polygons.is_empty() {
                    return None; // Skip if no polygons remain
                }
                geometry.value = geojson::Value::MultiPolygon(simplified_polygons);
            }
            geojson::Value::LineString(coords) => {
                if let Some(simplified_coords) = simplify_linestring(coords) {
                    geometry.value = geojson::Value::LineString(simplified_coords);
                } else {
                    return None;
                }
            }
            geojson::Value::MultiLineString(lines) => {
                let mut simplified_lines = Vec::new();
                for line in lines {
                    if let Some(simplified_line) = simplify_linestring(line) {
                        simplified_lines.push(simplified_line);
                    }
                }
                if simplified_lines.is_empty() {
                    return None;
                }
                geometry.value = geojson::Value::MultiLineString(simplified_lines);
            }
            _ => {} // Keep other geometry types as-is
        }
    }

    // Check if feature meets minimum area requirement
    let area = get_feature_area(&feature);
    if area < MIN_AREA_THRESHOLD {
        return None;
    }

    Some(feature)
}

/// Simplify polygon rings and filter by area
fn simplify_polygon_rings(rings: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
    let mut simplified_rings = Vec::new();
    
    for (i, ring) in rings.iter().enumerate() {
        if ring.len() < 4 {
            continue; // Skip invalid rings
        }

        // Calculate ring area (only for first ring - exterior)
        if i == 0 || calculate_ring_area(ring) >= MIN_RING_AREA {
            if let Some(simplified_ring) = simplify_ring(ring) {
                simplified_rings.push(simplified_ring);
            }
        }
    }
    
    simplified_rings
}

/// Simplify a single ring using Douglas-Peucker-like approach
fn simplify_ring(ring: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    if ring.len() < 4 {
        return None;
    }

    // Convert to geo-types for simplification
    let coords: Vec<Coord> = ring.iter()
        .map(|coord| Coord { x: coord[0], y: coord[1] })
        .collect();
    
    let linestring = LineString::from(coords);
    let simplified = linestring.simplify_vw(&SIMPLIFY_TOLERANCE);
    
    if simplified.coords().count() < 4 {
        return None;
    }

    // Convert back to GeoJSON format
    let simplified_ring: Vec<Vec<f64>> = simplified.coords()
        .map(|coord| vec![coord.x, coord.y])
        .collect();
    
    Some(simplified_ring)
}

/// Simplify a linestring
fn simplify_linestring(line: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    if line.len() < 2 {
        return None;
    }

    let coords: Vec<Coord> = line.iter()
        .map(|coord| Coord { x: coord[0], y: coord[1] })
        .collect();
    
    let linestring = LineString::from(coords);
    let simplified = linestring.simplify_vw(&SIMPLIFY_TOLERANCE);
    
    if simplified.coords().count() < 2 {
        return None;
    }

    let simplified_line: Vec<Vec<f64>> = simplified.coords()
        .map(|coord| vec![coord.x, coord.y])
        .collect();
    
    Some(simplified_line)
}

/// Calculate approximate area of a ring using shoelace formula
fn calculate_ring_area(ring: &[Vec<f64>]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    let n = ring.len();
    
    for i in 0..n {
        let j = (i + 1) % n;
        area += ring[i][0] * ring[j][1];
        area -= ring[j][0] * ring[i][1];
    }
    
    (area / 2.0).abs()
}

/// Get the area of a feature from its properties or calculate it
fn get_feature_area(feature: &geojson::Feature) -> f64 {
    // Try to get area from bbox if available
    if let Some(bbox) = &feature.bbox {
        if bbox.len() >= 4 {
            let width = bbox[2] - bbox[0];
            let height = bbox[3] - bbox[1];
            return width * height;
        }
    }

    // Fallback: calculate from geometry
    if let Some(geometry) = &feature.geometry {
        return calculate_geometry_area(&geometry.value);
    }

    0.0
}

/// Calculate area of a geometry
fn calculate_geometry_area(geometry: &geojson::Value) -> f64 {
    match geometry {
        geojson::Value::Polygon(rings) => {
            if !rings.is_empty() {
                calculate_ring_area(&rings[0])
            } else {
                0.0
            }
        }
        geojson::Value::MultiPolygon(polygons) => {
            polygons.iter()
                .map(|rings| if !rings.is_empty() { calculate_ring_area(&rings[0]) } else { 0.0 })
                .sum()
        }
        _ => 0.0,
    }
}

/// Round all coordinates in a geometry to reduce precision
fn round_geometry_coordinates(feature: &mut geojson::Feature) {
    if let Some(ref mut geometry) = feature.geometry {
        match &mut geometry.value {
            geojson::Value::Polygon(rings) => {
                for ring in rings {
                    for coord in ring {
                        coord[0] = round_coordinate(coord[0]);
                        coord[1] = round_coordinate(coord[1]);
                    }
                }
            }
            geojson::Value::LineString(coords) => {
                for coord in coords {
                    coord[0] = round_coordinate(coord[0]);
                    coord[1] = round_coordinate(coord[1]);
                }
            }
            geojson::Value::MultiLineString(lines) => {
                for line in lines {
                    for coord in line {
                        coord[0] = round_coordinate(coord[0]);
                        coord[1] = round_coordinate(coord[1]);
                    }
                }
            }
            geojson::Value::MultiPolygon(polygons) => {
                for polygon in polygons {
                    for ring in polygon {
                        for coord in ring {
                            coord[0] = round_coordinate(coord[0]);
                            coord[1] = round_coordinate(coord[1]);
                        }
                    }
                }
            }
            _ => {} // Keep other geometry types as-is
        }
    }
}

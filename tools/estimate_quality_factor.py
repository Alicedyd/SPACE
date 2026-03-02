#!/usr/bin/env python3
import argparse
import os
import json
import subprocess
import sys
from pathlib import Path

def estimate_jpeg_quality_imagemagick(image_path):
    """
    Estimate JPEG quality factor using ImageMagick's identify command.
    Returns quality factor as float, or None if cannot determine.
    """
    try:
        # Use ImageMagick's identify command to get quality
        # %Q format specifier returns the JPEG quality
        cmd = ['identify', '-format', '%Q', str(image_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            quality_str = result.stdout.strip()
            if quality_str and quality_str.isdigit():
                return float(quality_str)
            elif quality_str == '0':
                # Quality 0 usually means not a JPEG or no quality info
                return None
        
        # Fallback: try verbose output and parse
        cmd_verbose = ['identify', '-verbose', str(image_path)]
        result_verbose = subprocess.run(cmd_verbose, capture_output=True, text=True, timeout=10)
        
        if result_verbose.returncode == 0:
            lines = result_verbose.stdout.split('\n')
            for line in lines:
                if 'Quality:' in line:
                    # Extract quality value from line like "Quality: 90"
                    try:
                        quality_part = line.split('Quality:')[1].strip()
                        quality_num = quality_part.split()[0]
                        return float(quality_num)
                    except (IndexError, ValueError):
                        continue
        
        return None
        
    except subprocess.TimeoutExpired:
        print(f"Timeout processing {image_path}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"ImageMagick error for {image_path}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def check_imagemagick():
    """Check if ImageMagick is installed and accessible."""
    try:
        result = subprocess.run(['identify', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_info = result.stdout.split('\n')[0]
            print(f"Using {version_info}")
            return True
        else:
            return False
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract JPEG quality factors using ImageMagick')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('--output', '-o', default='quality_factors.json', 
                       help='Output JSON file (default: quality_factors.json)')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.jpg', '.jpeg', '.JPG', '.JPEG'],
                       help='Image file extensions to process')
    parser.add_argument('--include-non-jpeg', action='store_true',
                       help='Include non-JPEG files (will have None quality)')
    
    args = parser.parse_args()
    
    # Check if ImageMagick is available
    if not check_imagemagick():
        print("Error: ImageMagick is not installed or not accessible")
        print("Please install ImageMagick:")
        print("  Ubuntu/Debian: sudo apt-get install imagemagick")
        print("  macOS: brew install imagemagick")
        print("  Windows: Download from https://imagemagick.org/script/download.php")
        sys.exit(1)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)
    
    # Find all image files in the directory (depth=1)
    image_files = []
    for ext in args.extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        print(f"Looking for extensions: {args.extensions}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    quality_data = {}
    processed = 0
    jpeg_count = 0
    
    for image_path in sorted(image_files):
        basename = image_path.name
        quality = estimate_jpeg_quality_imagemagick(image_path)
        
        if quality is not None:
            quality_data[basename] = quality
            jpeg_count += 1
        elif args.include_non_jpeg:
            quality_data[basename] = None
        
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{len(image_files)} images")
    
    print(f"Successfully processed {processed} images")
    print(f"Found quality info for {jpeg_count} JPEG images")
    
    # Save to JSON file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(quality_data, f, indent=2, sort_keys=True)
    
    print(f"Quality factors saved to {output_path}")
    
    # Print some statistics
    jpeg_qualities = [q for q in quality_data.values() if q is not None]
    if jpeg_qualities:
        print(f"\nJPEG Quality Statistics:")
        print(f"  Images with quality info: {len(jpeg_qualities)}")
        print(f"  Average quality: {sum(jpeg_qualities)/len(jpeg_qualities):.1f}")
        print(f"  Min quality: {min(jpeg_qualities):.1f}")
        print(f"  Max quality: {max(jpeg_qualities):.1f}")
        
        # Quality distribution
        quality_ranges = [(0, 60), (60, 80), (80, 90), (90, 95), (95, 100)]
        print(f"  Quality distribution:")
        for low, high in quality_ranges:
            count = sum(1 for q in jpeg_qualities if low <= q < high)
            if low == 95:  # Last range includes 100
                count = sum(1 for q in jpeg_qualities if low <= q <= 100)
            print(f"    {low}-{high}: {count} images")

if __name__ == "__main__":
    main()
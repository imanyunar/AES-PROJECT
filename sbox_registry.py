import numpy as np
from image_crypto import test_image_sboxes

def extend_sbox_registry(sboxes: dict) -> dict:
    try:
        image_sboxes = test_image_sboxes()
        
        if not image_sboxes:
            print("⚠ Warning: No image S-boxes found in image_crypto.py")
            return sboxes
        
        added_count = 0
        
        for name, sbox in image_sboxes.items():
            try:
                if not isinstance(sbox, np.ndarray):
                    sbox = np.array(sbox, dtype=np.uint8)
                
                if len(sbox.shape) == 2:
                    if sbox.shape == (16, 16):
                        sbox = sbox.flatten()
                    else:
                        print(f"⚠ Warning: {name} has invalid 2D shape {sbox.shape}, expected (16, 16). Skipping.")
                        continue
                
                if len(sbox) != 256:
                    print(f"⚠ Warning: {name} has {len(sbox)} elements, expected 256. Skipping.")
                    continue
                
                if np.any(sbox < 0) or np.any(sbox > 255):
                    print(f"⚠ Warning: {name} contains values outside range 0-255. Skipping.")
                    continue
                
                unique_values = set(sbox)
                if len(unique_values) != 256:
                    print(f"⚠ Warning: {name} is not bijective (has {len(unique_values)} unique values, expected 256). Skipping.")
                    continue
                
                expected_values = set(range(256))
                if unique_values != expected_values:
                    missing = expected_values - unique_values
                    extra = unique_values - expected_values
                    print(f"⚠ Warning: {name} is not a valid permutation.")
                    if missing:
                        print(f"  Missing values: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}")
                    if extra:
                        print(f"  Extra values: {sorted(list(extra))[:10]}{'...' if len(extra) > 10 else ''}")
                    print("  Skipping.")
                    continue
                
                sbox = sbox.astype(np.uint8)
                
                registry_key = f"IMG-{name}"
                sboxes[registry_key] = sbox
                added_count += 1
                
            except Exception as e:
                print(f"⚠ Warning: Failed to process {name}: {e}. Skipping.")
                continue
        
        if added_count > 0:
            print(f"✓ Successfully added {added_count} image S-box(es) to registry")
        else:
            print("⚠ Warning: No image S-boxes were added (all failed validation)")
        
    except ImportError as e:
        print(f"⚠ Warning: Could not import test_image_sboxes from image_crypto.py: {e}")
        print("  Continuing with text S-boxes only...")
    except Exception as e:
        print(f"⚠ Warning: Unexpected error while loading image S-boxes: {e}")
        print("  Continuing with text S-boxes only...")
    
    return sboxes

def validate_sbox(sbox, name="Unknown"):
    try:
        if not isinstance(sbox, np.ndarray):
            return False, f"{name}: Not a numpy array"
        
        if sbox.shape not in [(256,), (16, 16)]:
            return False, f"{name}: Invalid shape {sbox.shape}, expected (256,) or (16, 16)"
        
        if len(sbox.shape) == 2:
            sbox = sbox.flatten()
        
        if len(sbox) != 256:
            return False, f"{name}: Invalid length {len(sbox)}, expected 256"
        
        if np.any(sbox < 0) or np.any(sbox > 255):
            return False, f"{name}: Contains values outside range 0-255"
        
        unique_values = set(sbox)
        if len(unique_values) != 256:
            return False, f"{name}: Not bijective (has {len(unique_values)} unique values)"
        
        expected_values = set(range(256))
        if unique_values != expected_values:
            return False, f"{name}: Not a valid permutation of 0-255"
        
        return True, None
        
    except Exception as e:
        return False, f"{name}: Validation error: {e}"

def list_available_sboxes(sboxes):
    print("=" * 80)
    print("AVAILABLE S-BOXES")
    print("=" * 80)
    
    if not sboxes:
        print("No S-boxes available")
        return
    
    print(f"Total: {len(sboxes)} S-boxes\n")
    
    text_sboxes = []
    image_sboxes = []
    other_sboxes = []
    
    for name, sbox in sboxes.items():
        if name.startswith("IMG-"):
            image_sboxes.append(name)
        elif name in ["AES", "A0", "A1", "A2", "K4", "K44", "K128"]:
            text_sboxes.append(name)
        else:
            other_sboxes.append(name)
    
    if text_sboxes:
        print("Text S-boxes (Affine Matrix Based):")
        for name in sorted(text_sboxes):
            sbox = sboxes[name]
            is_valid, _ = validate_sbox(sbox, name)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {name:<10} - {len(sbox)} elements")
        print()
    
    if image_sboxes:
        print("Image S-boxes (From Alamsyah et al. 2023):")
        for name in sorted(image_sboxes):
            sbox = sboxes[name]
            is_valid, _ = validate_sbox(sbox, name)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {name:<15} - {len(sbox)} elements")
        print()
    
    if other_sboxes:
        print("Other S-boxes:")
        for name in sorted(other_sboxes):
            sbox = sboxes[name]
            is_valid, _ = validate_sbox(sbox, name)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {name:<10} - {len(sbox)} elements")
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("S-BOX REGISTRY EXTENSION - TEST")
    print("=" * 80 + "\n")
    
    try:
        from sbox_utils import generate_sboxes
    except ImportError:
        print("Error: Could not import generate_sboxes from sbox_utils.py")
        exit(1)
    
    print("Step 1: Generating base S-boxes...")
    sboxes = generate_sboxes(include_random=False)
    print(f"  Generated {len(sboxes)} text S-boxes: {list(sboxes.keys())}\n")
    
    print("Step 2: Extending with image S-boxes...")
    sboxes = extend_sbox_registry(sboxes)
    print()
    
    print("Step 3: Listing all available S-boxes...")
    list_available_sboxes(sboxes)
    
    print("\nStep 4: Validating all S-boxes...")
    print("-" * 80)
    all_valid = True
    for name, sbox in sboxes.items():
        is_valid, error = validate_sbox(sbox, name)
        if is_valid:
            print(f"✓ {name:<15} - Valid")
        else:
            print(f"✗ {name:<15} - Invalid: {error}")
            all_valid = False
    
    print("-" * 80)
    if all_valid:
        print("✓ All S-boxes are valid!")
    else:
        print("⚠ Some S-boxes failed validation")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80 + "\n")
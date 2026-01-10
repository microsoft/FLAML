#!/usr/bin/env python3
"""
Test to verify that BlendSearch no longer produces OptunaSearch warnings
about unresolved search space definitions when using Ray Tune domains.
"""

import sys
import io
from contextlib import redirect_stderr

def test_blendsearch_no_warning_with_ray_tune_domains():
    """Test that BlendSearch doesn't produce warning with Ray Tune domains."""
    from ray import tune
    from flaml.tune.searcher.blendsearch import BlendSearch
    
    # Create a non-hierarchical space with Ray Tune domains
    config = {
        'lr': tune.uniform(0.001, 0.1),
        'max_depth': tune.randint(1, 10),
        'batch_size': tune.choice([16, 32, 64])
    }
    
    # Capture stderr to check for warnings
    stderr_capture = io.StringIO()
    with redirect_stderr(stderr_capture):
        searcher = BlendSearch(
            metric='loss',
            mode='min',
            space=config,
        )
    
    stderr_output = stderr_capture.getvalue()
    
    # Check that the warning is NOT present
    assert "unresolved search space" not in stderr_output, \
        f"Warning found in stderr: {stderr_output}"
    
    # Verify that the searcher works correctly
    assert searcher._gs is not None, "Global search should be initialized"
    assert callable(searcher._gs._space), "Global search space should be a callable (define-by-run)"
    
    # Test that suggest works
    trial_id = 'test_trial_1'
    config_suggested = searcher.suggest(trial_id)
    assert config_suggested is not None, "Should suggest a config"
    assert 'lr' in config_suggested, "Suggested config should have 'lr'"
    assert 'max_depth' in config_suggested, "Suggested config should have 'max_depth'"
    assert 'batch_size' in config_suggested, "Suggested config should have 'batch_size'"
    
    print("✓ Test passed: BlendSearch with Ray Tune domains produces no warning")


def test_blendsearch_no_warning_with_flaml_domains():
    """Test that BlendSearch doesn't produce warning with FLAML domains."""
    from flaml.tune import sample
    from flaml.tune.searcher.blendsearch import BlendSearch
    
    # Create a non-hierarchical space with FLAML domains
    config = {
        'lr': sample.uniform(0.001, 0.1),
        'max_depth': sample.randint(1, 10),
        'batch_size': sample.choice([16, 32, 64])
    }
    
    # Capture stderr to check for warnings
    stderr_capture = io.StringIO()
    with redirect_stderr(stderr_capture):
        searcher = BlendSearch(
            metric='loss',
            mode='min',
            space=config,
        )
    
    stderr_output = stderr_capture.getvalue()
    
    # Check that the warning is NOT present
    assert "unresolved search space" not in stderr_output, \
        f"Warning found in stderr: {stderr_output}"
    
    # Test that suggest works
    trial_id = 'test_trial_1'
    config_suggested = searcher.suggest(trial_id)
    assert config_suggested is not None, "Should suggest a config"
    
    print("✓ Test passed: BlendSearch with FLAML domains produces no warning")


def test_cfo_still_works():
    """Test that CFO still works as expected (doesn't use global search)."""
    from ray import tune
    from flaml.tune.searcher.blendsearch import CFO
    
    config = {
        'lr': tune.uniform(0.001, 0.1),
        'max_depth': tune.randint(1, 10),
    }
    
    searcher = CFO(
        metric='loss',
        mode='min',
        space=config,
    )
    
    # CFO should not have a global search
    assert searcher._gs is None, "CFO should not use global search"
    
    # Test that suggest works
    trial_id = 'test_trial_1'
    config_suggested = searcher.suggest(trial_id)
    assert config_suggested is not None, "Should suggest a config"
    
    print("✓ Test passed: CFO works correctly without global search")


if __name__ == '__main__':
    print("Testing BlendSearch fix for OptunaSearch warning...")
    print()
    
    try:
        test_blendsearch_no_warning_with_ray_tune_domains()
        print()
        test_blendsearch_no_warning_with_flaml_domains()
        print()
        test_cfo_still_works()
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

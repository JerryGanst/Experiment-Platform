#!/usr/bin/env python3
"""
Unit tests and integration tests for HACE Diagnostic System
"""

import os
import sys
import json
import tempfile
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDiagnosticRecorder(unittest.TestCase):
    """Unit tests for DiagnosticRecorder class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Enable diagnostic mode
        os.environ["HACE_DIAGNOSTIC"] = "1"
        # Import after setting env var
        from cake.model.modify_qwen2 import DiagnosticRecorder
        self.recorder = DiagnosticRecorder()
        self.recorder.enabled = True  # Force enable
        
    def tearDown(self):
        """Clean up"""
        os.environ.pop("HACE_DIAGNOSTIC", None)
        
    def test_record_single_entry(self):
        """Test recording a single entry"""
        self.recorder.record(
            layer_idx=0,
            head_idx=0,
            kept_indices=[1, 2, 3, 4, 5],
            head_budget=5,
            head_entropy=2.5,
            total_seq_len=100
        )
        
        self.assertEqual(len(self.recorder.records), 1)
        record = self.recorder.records[0]
        self.assertEqual(record['layer_idx'], 0)
        self.assertEqual(record['head_idx'], 0)
        self.assertEqual(record['kept_indices'], [1, 2, 3, 4, 5])
        self.assertEqual(record['head_budget'], 5)
        self.assertAlmostEqual(record['head_entropy'], 2.5)
        self.assertAlmostEqual(record['kept_ratio'], 0.05)
        
    def test_record_multiple_entries(self):
        """Test recording multiple entries"""
        for layer in range(3):
            for head in range(4):
                self.recorder.record(
                    layer_idx=layer,
                    head_idx=head,
                    kept_indices=list(range(10)),
                    head_budget=10,
                    head_entropy=1.0 + layer * 0.1,
                    total_seq_len=100
                )
        
        self.assertEqual(len(self.recorder.records), 12)
        
    def test_next_sample(self):
        """Test sample ID incrementing"""
        self.assertEqual(self.recorder.sample_id, 0)
        
        self.recorder.record(layer_idx=0, head_idx=0, kept_indices=[1], 
                            head_budget=1, head_entropy=1.0, total_seq_len=10)
        
        self.recorder.next_sample()
        self.assertEqual(self.recorder.sample_id, 1)
        
        self.recorder.record(layer_idx=0, head_idx=0, kept_indices=[2], 
                            head_budget=1, head_entropy=1.0, total_seq_len=10)
        
        # Check that sample_id is correctly recorded
        self.assertEqual(self.recorder.records[0]['sample_id'], 0)
        self.assertEqual(self.recorder.records[1]['sample_id'], 1)
        
    def test_save_and_load(self):
        """Test saving to JSON file"""
        self.recorder.record(
            layer_idx=5,
            head_idx=3,
            kept_indices=[10, 20, 30],
            head_budget=3,
            head_entropy=3.14,
            total_seq_len=200
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            
        try:
            self.recorder.save(temp_path)
            
            # Load and verify
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
                
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]['layer_idx'], 5)
            self.assertEqual(loaded[0]['head_idx'], 3)
            self.assertEqual(loaded[0]['kept_indices'], [10, 20, 30])
        finally:
            os.unlink(temp_path)
            
    def test_clear(self):
        """Test clearing records"""
        self.recorder.record(layer_idx=0, head_idx=0, kept_indices=[1], 
                            head_budget=1, head_entropy=1.0, total_seq_len=10)
        self.recorder.next_sample()
        
        self.assertEqual(len(self.recorder.records), 1)
        self.assertEqual(self.recorder.sample_id, 1)
        
        self.recorder.clear()
        
        self.assertEqual(len(self.recorder.records), 0)
        self.assertEqual(self.recorder.sample_id, 0)
        
    def test_get_stats(self):
        """Test statistics gathering"""
        # Empty stats
        self.assertEqual(self.recorder.get_stats(), {})
        
        # Add some records
        for layer in range(2):
            for head in range(3):
                self.recorder.record(
                    layer_idx=layer, head_idx=head,
                    kept_indices=[1, 2], head_budget=2,
                    head_entropy=1.0, total_seq_len=10
                )
        
        stats = self.recorder.get_stats()
        self.assertEqual(stats['num_records'], 6)
        self.assertEqual(stats['num_samples'], 1)
        self.assertEqual(stats['num_layers'], 2)
        self.assertEqual(stats['num_heads'], 3)
        
    def test_disabled_mode(self):
        """Test that recording is skipped when disabled"""
        self.recorder.enabled = False
        
        self.recorder.record(layer_idx=0, head_idx=0, kept_indices=[1], 
                            head_budget=1, head_entropy=1.0, total_seq_len=10)
        
        self.assertEqual(len(self.recorder.records), 0)


class TestDiagnosticVisualize(unittest.TestCase):
    """Unit tests for diagnostic_visualize.py functions"""
    
    def setUp(self):
        """Create test data files"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create baseline diagnostic data
        self.baseline_data = [
            {"sample_id": 0, "layer_idx": 0, "head_idx": 0, 
             "kept_indices": [1, 2, 3, 4, 5], "head_budget": 5, 
             "head_entropy": 0.0, "total_seq_len": 100, "kept_ratio": 0.05},
            {"sample_id": 0, "layer_idx": 0, "head_idx": 1, 
             "kept_indices": [1, 2, 3, 4, 5], "head_budget": 5, 
             "head_entropy": 0.0, "total_seq_len": 100, "kept_ratio": 0.05},
        ]
        
        # Create Ada diagnostic data (different indices)
        self.ada_data = [
            {"sample_id": 0, "layer_idx": 0, "head_idx": 0, 
             "kept_indices": [1, 2, 3, 6, 7], "head_budget": 5, 
             "head_entropy": 2.5, "total_seq_len": 100, "kept_ratio": 0.05},
            {"sample_id": 0, "layer_idx": 0, "head_idx": 1, 
             "kept_indices": [2, 3, 4, 8, 9], "head_budget": 5, 
             "head_entropy": 1.8, "total_seq_len": 100, "kept_ratio": 0.05},
        ]
        
        self.baseline_path = os.path.join(self.temp_dir, "baseline.json")
        self.ada_path = os.path.join(self.temp_dir, "ada.json")
        
        with open(self.baseline_path, 'w') as f:
            json.dump(self.baseline_data, f)
        with open(self.ada_path, 'w') as f:
            json.dump(self.ada_data, f)
            
    def tearDown(self):
        """Clean up temp files"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_load_diagnostic(self):
        """Test loading diagnostic files"""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from diagnostic_visualize import load_diagnostic
        
        data = load_diagnostic(self.baseline_path)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['layer_idx'], 0)
        
    def test_compare_two_runs(self):
        """Test comparison function"""
        from diagnostic_visualize import compare_two_runs
        
        head_diffs, seq_len = compare_two_runs(
            self.baseline_path, self.ada_path,
            sample_id=0, layer_idx=0
        )
        
        self.assertIsNotNone(head_diffs)
        self.assertEqual(len(head_diffs), 2)
        self.assertEqual(seq_len, 100)
        
        # Check head 0: baseline=[1,2,3,4,5], ada=[1,2,3,6,7]
        # common=[1,2,3], only_baseline=[4,5], only_ada=[6,7]
        h0 = head_diffs[0]
        self.assertEqual(h0['common_count'], 3)
        self.assertEqual(sorted(h0['only_baseline']), [4, 5])
        self.assertEqual(sorted(h0['only_ada']), [6, 7])
        
        # Jaccard = 3 / 7 = 0.4286
        self.assertAlmostEqual(h0['jaccard'], 3/7, places=3)
        
    def test_analyze_answer_overlap(self):
        """Test answer overlap analysis"""
        from diagnostic_visualize import compare_two_runs, analyze_answer_overlap
        
        head_diffs, _ = compare_two_runs(
            self.baseline_path, self.ada_path,
            sample_id=0, layer_idx=0
        )
        
        # Answer at positions 4, 5 (which baseline kept but Ada dropped)
        result = analyze_answer_overlap(head_diffs, [4, 5])
        
        self.assertGreater(result['dropped'], 0)
        
    def test_missing_sample(self):
        """Test handling of missing sample"""
        from diagnostic_visualize import compare_two_runs
        
        head_diffs, seq_len = compare_two_runs(
            self.baseline_path, self.ada_path,
            sample_id=999, layer_idx=0
        )
        
        self.assertIsNone(head_diffs)
        self.assertEqual(seq_len, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the diagnostic workflow"""
    
    def test_full_workflow(self):
        """Test complete diagnostic workflow without actual model"""
        import tempfile
        import json
        
        os.environ["HACE_DIAGNOSTIC"] = "1"
        
        from cake.model.modify_qwen2 import DIAGNOSTIC
        DIAGNOSTIC.clear()
        DIAGNOSTIC.enabled = True
        
        # Simulate baseline recording (all heads share same indices)
        num_layers = 3
        num_heads = 4
        seq_len = 100
        budget = 10
        
        for layer in range(num_layers):
            shared_indices = list(range(budget))  # [0, 1, ..., 9]
            for head in range(num_heads):
                DIAGNOSTIC.record(
                    layer_idx=layer,
                    head_idx=head,
                    kept_indices=shared_indices,
                    head_budget=budget,
                    head_entropy=0.0,
                    total_seq_len=seq_len
                )
        
        # Save baseline
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_path = f.name
        DIAGNOSTIC.save(baseline_path)
        
        # Clear and simulate Ada recording (different indices per head)
        DIAGNOSTIC.clear()
        DIAGNOSTIC.enabled = True
        
        for layer in range(num_layers):
            for head in range(num_heads):
                # Each head has slightly different indices
                head_indices = list(range(head, budget + head))
                DIAGNOSTIC.record(
                    layer_idx=layer,
                    head_idx=head,
                    kept_indices=head_indices,
                    head_budget=budget,
                    head_entropy=1.0 + head * 0.5,  # Different entropy per head
                    total_seq_len=seq_len
                )
        
        # Save Ada
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            ada_path = f.name
        DIAGNOSTIC.save(ada_path)
        
        # Run comparison
        from diagnostic_visualize import compare_two_runs, compare_across_layers
        
        head_diffs, result_seq_len = compare_two_runs(
            baseline_path, ada_path,
            sample_id=0, layer_idx=0
        )
        
        self.assertIsNotNone(head_diffs)
        self.assertEqual(len(head_diffs), num_heads)
        self.assertEqual(result_seq_len, seq_len)
        
        # Verify head 0 has same indices (baseline=[0..9], ada=[0..9])
        self.assertEqual(head_diffs[0]['jaccard'], 1.0)
        
        # Verify head 3 has different indices (baseline=[0..9], ada=[3..12])
        # common=[3..9]=7, total=[0..12]=13, jaccard=7/13
        h3 = head_diffs[3]
        self.assertAlmostEqual(h3['jaccard'], 7/13, places=3)
        
        # Clean up
        os.unlink(baseline_path)
        os.unlink(ada_path)
        os.environ.pop("HACE_DIAGNOSTIC", None)
        
    def test_visualization_output(self):
        """Test that visualization generates output file"""
        import tempfile
        import json
        
        temp_dir = tempfile.mkdtemp()
        
        # Create test data
        baseline_data = [
            {"sample_id": 0, "layer_idx": 0, "head_idx": h, 
             "kept_indices": list(range(10)), "head_budget": 10, 
             "head_entropy": 0.0, "total_seq_len": 100, "kept_ratio": 0.1}
            for h in range(4)
        ]
        ada_data = [
            {"sample_id": 0, "layer_idx": 0, "head_idx": h, 
             "kept_indices": list(range(h, 10+h)), "head_budget": 10, 
             "head_entropy": 1.0 + h * 0.3, "total_seq_len": 100, "kept_ratio": 0.1}
            for h in range(4)
        ]
        
        baseline_path = os.path.join(temp_dir, "baseline.json")
        ada_path = os.path.join(temp_dir, "ada.json")
        output_path = os.path.join(temp_dir, "output.png")
        
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f)
        with open(ada_path, 'w') as f:
            json.dump(ada_data, f)
        
        # Generate visualization
        from diagnostic_visualize import compare_two_runs, visualize_token_retention
        
        head_diffs, seq_len = compare_two_runs(baseline_path, ada_path, 0, 0)
        
        # This should not raise an error
        try:
            visualize_token_retention(head_diffs, seq_len, output_path)
            # Check file was created
            self.assertTrue(os.path.exists(output_path))
        except Exception as e:
            # matplotlib might not have display, that's ok
            if "display" not in str(e).lower() and "backend" not in str(e).lower():
                raise
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)

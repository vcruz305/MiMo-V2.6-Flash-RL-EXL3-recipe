"""Recompute completed round5/6 results from tracked records; no GPU/network."""
import json
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]


class QuantRecordTests(unittest.TestCase):
    def test_records_and_recomputed_statistics(self):
        records = []
        expected_order = [(case, rep) for rep in range(4) for case in ('code', 'prose')]
        for filename, tags, count in (
            ('france-round5-quant-drafter.json', ['bf16', 'quant'], 90),
            ('france-round6-quant-readback.json', ['A', 'B', 'A2'], 65),
        ):
            data = json.loads((ROOT / 'bench' / filename).read_text(encoding='utf-8'))
            records.append(data)
            self.assertEqual(list(data['arms']), tags)
            self.assertEqual(data['provenance']['verified_source_manifest_entries'], count)
            self.assertEqual(data['protocol']['seed'], 42)
            self.assertEqual(data['protocol']['temperature'], 0)
            self.assertEqual(data['protocol']['concurrency'], 1)
            self.assertEqual(data['drafter']['revision'], 'd50ead3c6a3dec221e9a595fbdc103ef60db594e')
            for tag, arm in data['arms'].items():
                rows = arm['responses']
                self.assertEqual([(r['case'], r['rep']) for r in rows], expected_order)
                for row in rows:
                    self.assertEqual(row['cap'], 320)
                    self.assertEqual(row['phase'], 'cold' if row['rep'] == 0 else 'warm')
                for case in ('code', 'prose'):
                    warm = [r for r in rows if r['case'] == case and r['rep'] > 0]
                    self.assertEqual(len(warm), 3)
                    for field, key in [('decode_tok_s', 'decode_tok_s_mean'), ('draft_accept', 'draft_accept_mean')]:
                        self.assertAlmostEqual(statistics.mean(r['usage'][field] for r in warm), arm['warm'][case][key])
                command = arm['launch']['command']
                for flag, value in {'-cs': '4096', '-cq': '4', '-chunk_size': '4096', '-ambs': '1', '-ndt': '7', '-dc': '0.6', '-gs': '106', '--max-active-requests': '1'}.items():
                    self.assertEqual(command[command.index(flag) + 1], value)
                self.assertIn('-dds', command)
                self.assertEqual(arm['launch']['env']['EXL3_BATCH_VERIFY'], '0')
                self.assertEqual(arm['launch']['env']['EXL3_UMA_RESERVE_MB'], '8192')
                self.assertEqual(arm['launch']['max_seconds'], 43200)
            for case, result in data['comparison'].items():
                ref = statistics.mean(data['arms'][tag]['warm'][case]['decode_tok_s_mean'] for tag in result['reference_tags'])
                candidate = data['arms'][result['candidate_tag']]['warm'][case]['decode_tok_s_mean']
                self.assertAlmostEqual(ref, result['reference_mean'])
                self.assertAlmostEqual((candidate / ref - 1) * 100, result['gain_percent'])
            for prefill in data['fresh_prefill'].values():
                self.assertTrue(prefill['correct'])
                self.assertEqual(prefill['text'], prefill['expected'])
                self.assertEqual(prefill['usage']['prompt_tokens'], 3527)
                self.assertAlmostEqual(3527 / prefill['usage']['time_prefill'], prefill['prefill_tok_s'])
        r5, r6 = records
        matched = []
        for a, b in zip(r5['arms']['bf16']['responses'], r5['arms']['quant']['responses']):
            self.assertEqual(a['prompt'], b['prompt'])
            self.assertNotEqual(a['usage']['draft_accept'], b['usage']['draft_accept'])
            if a['text'] == b['text']:
                matched.append((a['case'], a['rep']))
                self.assertEqual(a['usage']['completion_tokens'], b['usage']['completion_tokens'])
        self.assertEqual(matched, [('code', 1), ('code', 2), ('code', 3)])
        for a, b, a2 in zip(*(r6['arms'][tag]['responses'] for tag in ('A', 'B', 'A2'))):
            for other in (b, a2):
                self.assertEqual(a['prompt'], other['prompt'])
                self.assertEqual(a['text'], other['text'])
                for field in ('completion_tokens', 'draft_accept'):
                    self.assertEqual(a['usage'][field], other['usage'][field])
        self.assertEqual(r6['arms']['B']['launch']['env']['EXL3_MOE_MIXEDK_ELIDE_HANDLED'], '1')
        for tag in ('A', 'A2'):
            self.assertNotIn('EXL3_MOE_MIXEDK_ELIDE_HANDLED', r6['arms'][tag]['launch']['env'])
        print('Verified 40 ordered responses; round5 3/8 text matches; round6 8/8 text/count/acceptance matches; 6 fresh prefills.')
        print(json.dumps({r['measurement']: r['comparison'] for r in records}, indent=2))


if __name__ == '__main__':
    unittest.main()

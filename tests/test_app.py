import unittest
from unittest.mock import patch
from PIL import Image
import app

class RegressionTests(unittest.TestCase):
    def test_no_image(self):
        with self.assertRaises(app.gr.Error): app.analyze_image(None)

    def test_readable_boxes(self):
        for box in (app.caption_output,app.emotion_output,app.environment_output):
            self.assertGreaterEqual(box.lines,5)
            self.assertGreaterEqual(box.max_lines,14)

    def test_no_person_skips_face_detector(self):
        with patch.object(app,'face_detector',side_effect=AssertionError('Must not run')):
            self.assertIn('no person',app.detect_emotion(Image.new('RGB',(80,80)),[]))

    def test_green_does_not_claim_mountains(self):
        env=app.detect_environment('a parrot',Image.new('RGB',(80,80),(20,120,20)))
        self.assertIn('Green surroundings',env)
        self.assertNotIn('Mountain',env)

    def test_caption_is_not_a_story(self):
        self.assertFalse(app.story_is_complete('A parrot on a branch.','a parrot',['bird']))

    def test_failed_model_still_provides_labelled_story(self):
        with patch.object(app,'text_components',side_effect=RuntimeError('offline')):
            story=app.generate_story('a parrot on a branch',['bird'])
        self.assertIn('template-assisted fallback',story)
        self.assertGreater(len(story.split()),100)
        self.assertEqual(len(story.split('\n\n')),4)

    def test_description_paragraphs(self):
        text=app.generate_scene_description('a green parrot',['bird'],'No person','Green surroundings')
        self.assertEqual(len(text.split('\n\n')),3)
        self.assertIn('parrot',text)

if __name__=='__main__': unittest.main()

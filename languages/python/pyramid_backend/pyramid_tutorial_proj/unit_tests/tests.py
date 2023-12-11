import unittest

from pyramid import testing


class TutorialViewTests(unittest.TestCase):
    def setUp(self):
        self.config = testing.setUp()

    def tearDown(self):
        testing.tearDown()

    def test_home(self):
        from pyramid_tutorial.views.views import home

        request = testing.DummyRequest()
        request.headers["Host"] = "DummyGuest:2222"
        response = home(request)
        self.assertEqual(response["name"], request.headers["Host"])
        

class TutorialFunctionalTests(unittest.TestCase):
    def setUp(self):
        from pyramid_tutorial import main
        app = main({})
        from webtest import TestApp

        self.testapp = TestApp(app)

    def test_hello_world(self):
        response = self.testapp.get('/', status=200)
        self.assertEqual(response.status_code, 200)

    def test_hello(self):
        response = self.testapp.get('/countVisits', status=200)
        self.assertEqual(response.status_code, 200)

    def test_css(self):
        response = self.testapp.get('/static/css/app.css', status=200)
        self.assertEqual(response.status_code, 200)

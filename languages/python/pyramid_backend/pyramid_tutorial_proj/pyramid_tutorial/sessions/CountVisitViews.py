from pyramid.view import (
    view_config,
    view_defaults
    )

project_name = "pyramid_tutorial"


@view_defaults(renderer=project_name + ':' + 'templates/countVisits.pt')
class CountVisitViews:
    def __init__(self, request):
        self.request = request

    @property
    def counter(self):
        session = self.request.session
        if 'counter' in session:
            session['counter'] += 1
        else:
            session['counter'] = 1

        return session['counter']


    @view_config(route_name='countVisits')
    def countVisits(self):
        return {}

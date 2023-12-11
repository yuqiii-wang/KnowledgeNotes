from pyramid.config import Configurator
from pyramid.session import SignedCookieSessionFactory


def main(global_config, **settings):
    my_session_factory = SignedCookieSessionFactory(
        'itsaseekreet')
    config = Configurator(settings=settings,
                          session_factory=my_session_factory)
    config.include('pyramid_chameleon')
    config.add_route('home', '/')
    config.add_route('countVisits', '/countVisits')
    config.add_static_view(name='static', path='pyramid_tutorial:static')
    config.scan('.views')
    config.scan('.templates')
    config.scan('.sessions')
    config.scan('.static')
    return config.make_wsgi_app()
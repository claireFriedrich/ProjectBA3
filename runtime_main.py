import cProfile
import pstats
from main import main


cProfile.run('main()', 'summary.stats')
stats = pstats.Stats('summary.stats')
stats.sort_stats('calls')
stats.print_stats()



from net_work import trainner

import setproctitle
setproctitle.setproctitle("blrcnn*_*_")


trainner=trainner()

trainner.train()

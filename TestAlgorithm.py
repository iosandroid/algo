#!/usr/bin/env python
#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from zipline.algorithm import TradingAlgorithm



class TestAlgorithm(TradingAlgorithm):
    """
    This algorithm will send a specified number of orders, to allow unit tests
    to verify the orders sent/received, transactions created, and positions
    at the close of a simulation.
    """

    def initialize(context):
        context.i = 0
        context.asset = symbol('EURUSD1')

        pass

    def handle_data(self, data):    
        short_mavg = data.history(context.asset, 'price', bar_count=10, frequency="1m").mean()
        long_mavg = data.history(context.asset, 'price', bar_count=20, frequency="1m").mean()

        # Trading logic
        if short_mavg > long_mavg:
            # order_target orders as many shares as needed to
            # achieve the desired number of shares.
            order_target(context.asset, 100)
        elif short_mavg < long_mavg:
            order_target(context.asset, 0)

        # Save values for later inspection        
        record(EURUSD1=data.current(context.asset, 'price'),
           short_mavg=short_mavg,
           long_mavg=long_mavg)

        pass
    
    def analyze(context, perf):
        fig = plt.figure()
    
        ax1 = fig.add_subplot(311)    
        perf.portfolio_value.plot(ax=ax1)        
        ax1.set_ylabel('portfolio value in $')

        ax2 = fig.add_subplot(312)
    
    
        perf.EURUSD1.plot(ax=ax2)
        perf.short_mavg.plot(ax=ax2)
        perf.long_mavg.plot(ax=ax2)

        perf_trans = perf.ix[[t != [] for t in perf.transactions]]
    
        buys = perf_trans.ix[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
        sells = perf_trans.ix[[t[0]['amount'] < 0 for t in perf_trans.transactions]]
    
        ax2.plot(buys.index, perf.short_mavg.ix[buys.index],'^', markersize=10, color='m')
        ax2.plot(sells.index, perf.short_mavg.ix[sells.index],'v', markersize=10, color='k')
    
        ax2.set_ylabel('price in $')
        plt.legend(loc=0)
    
        ax3 = fig.add_subplot(313)
        perf.algorithm_period_return.plot(ax=ax3)
    
        plt.gcf().set_size_inches(18, 12)
        plt.legend(loc=0)
        plt.show()

        pass


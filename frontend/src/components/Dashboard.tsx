import React from 'react';
import { MarketDataChart } from './charts';
import { AnalysisPanel } from './analysis';

export const Dashboard: React.FC = () => {
    return (
        <div>
            <MarketDataChart />
            <AnalysisPanel />
        </div>
    );
}; 
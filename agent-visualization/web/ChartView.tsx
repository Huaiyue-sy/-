import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';

export interface ChartHandle {
  getInstance(): echarts.ECharts | null;
}

const ChartView = forwardRef<ChartHandle, { option: EChartsOption }>(({ option }, ref) => {
  const container = useRef<HTMLDivElement>(null);
  const instance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!container.current) return;
    const chart = echarts.init(container.current);
    instance.current = chart;
    chart.setOption(option, { notMerge: true });
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(container.current);
    return () => {
      observer.disconnect();
      chart.dispose();
      instance.current = null;
    };
  }, []);

  useEffect(() => {
    instance.current?.setOption(option, { notMerge: true, lazyUpdate: true });
  }, [option]);

  useImperativeHandle(ref, () => ({ getInstance: () => instance.current }), []);

  return <div ref={container} className="chart-canvas" />;
});

ChartView.displayName = 'ChartView';

export default ChartView;

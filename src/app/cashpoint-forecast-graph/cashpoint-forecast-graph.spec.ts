import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CashpointForecastGraph } from './cashpoint-forecast-graph';

describe('CashpointForecastGraph', () => {
  let component: CashpointForecastGraph;
  let fixture: ComponentFixture<CashpointForecastGraph>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CashpointForecastGraph]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CashpointForecastGraph);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
